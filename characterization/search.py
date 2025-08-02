import argparse
import requests
import urllib.parse
from datetime import datetime
from abc import ABC, abstractmethod
from xml.etree import ElementTree as ET

import arxiv
from utils import save_to_csv, remove_duplicate_titles, filter_by_keyword

SUPPORTING_DB = ["arxiv", "pubmed"]


class BaseSearcher(ABC):
    @abstractmethod
    def build_query(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(self, *args, **kwargs):
        pass

    @abstractmethod
    def parse_metadata(self, *args, **kwargs):
        pass


class ArxivSearcher(BaseSearcher):
    def __init__(self):
        super().__init__()

    def build_query(self, topic, start_year=None, end_year=None):
        current_year = datetime.today().year
        if end_year is None: end_year = current_year
        if start_year is None: start_year = current_year - 5
        date_from = f"{start_year}0101"
        date_to = f"{end_year}1231"
        return f"{topic} AND submittedDate:[{date_from} TO {date_to}]"

    def parse_metadata(self, search_results):
        metadata = []
        for i, result in enumerate(search_results):
            title = result.title
            authors = ", ".join([a.name for a in result.authors])
            year = result.updated.year
            url = result.entry_id
            abstract = result.summary
            metadata.append({
                "title": str(title),
                "authors": str(authors),
                "year": int(year),
                "url": str(url),
                "abstract": str(abstract)
            })
        return metadata

    def search(self,
               topic,
               max_results=5,
               start_year=None,
               end_year=None):

        # Build query
        query = self.build_query(
            topic=topic,
            start_year=start_year,
            end_year=end_year
        )
        print(f"🔍 Searching {max_results} article(s) on arXiv with query: '{query}'...\n")

        # Search articles
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return client.results(search)


class PubMedSearcher(BaseSearcher):
    def __init__(self):
        super().__init__()

    def build_query(self, topic):
        return f"{topic}[Title]"

    def search(self, topic, max_results=5):
        encoded_topic = urllib.parse.quote_plus(topic)
        query = self.build_query(encoded_topic)
        print(f"🔍 Searching {max_results} article(s) on pubmed with query: '{query}'...\n")

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=xml"
        response = requests.get(url)
        return response.content

    def parse_metadata(self, search_results):
        root = ET.fromstring(search_results)
        pmids = [id.text for id in root.findall('.//Id')]
        
        if not pmids:
            return []

        efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
        efetch_response = requests.get(efetch_url)
        efetch_root = ET.fromstring(efetch_response.content)

        metadata = []
        for article in efetch_root.findall('.//PubmedArticle'):
            title = article.find('.//ArticleTitle').text
            authors = ", ".join([author.find('LastName').text + " " + author.find('ForeName').text for author in article.findall('.//Author')])
            year = article.find('.//PubDate/Year').text
            pmid = article.find('.//PMID').text
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            abstract = article.find('.//AbstractText').text if article.find('.//AbstractText') is not None else ""
            metadata.append({
                "title": str(title),
                "authors": str(authors),
                "year": int(year),
                "url": str(url),
                "abstract": str(abstract)
            })
        return metadata


def process_metadata(metadata, output_csv, keyword=None):
    filtered_metadata = remove_duplicate_titles(metadata) if len(metadata) > 1 else metadata
    print(f"Found {len(filtered_metadata)} unique articles.")

    if keyword:
        for k in keyword:
            filtered_metadata = filter_by_keyword(
                metadata=filtered_metadata,
                keyword=k
            )
            print(f"Found {len(filtered_metadata)} articles containing keyword '{k}'.")

    if len(filtered_metadata) > 0:
        save_to_csv(metadata=filtered_metadata, output_csv=output_csv)
        print(f"Save {len(filtered_metadata)} search results to {output_csv}")


def search_articles(args):
    db_name = args.db
    db_name = db_name.lower().strip()

    if db_name == "arxiv":
        searcher = ArxivSearcher()
        results = searcher.search(
            topic=args.topic,
            max_results=args.max_results,
            start_year=args.start_year,
            end_year=args.end_year
        )
        metadata = searcher.parse_metadata(results)
        process_metadata(
            metadata=metadata,
            output_csv=args.output_csv,
            keyword=args.keyword
        )

    elif db_name == "pubmed":
        searcher = PubMedSearcher()
        results  = searcher.search(topic=args.topic, max_results=args.max_results)
        metadata = searcher.parse_metadata(results)
        process_metadata(
            metadata=metadata,
            output_csv=args.output_csv,
            keyword=args.keyword
        )
    else:
        NotImplementedError(f"Invalid database name {db_name}. Supported values are: {SUPPORTING_DB}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search articles on arXiv.")
    parser.add_argument("topic", type=str, help="Search topic")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of papers to fetch")
    parser.add_argument("--start-year", type=int, default=None, help="Start year for filtering papers")
    parser.add_argument("--end-year", type=int, default=None, help="End year for filtering papers")
    parser.add_argument("--db", type=str, default="arxiv", help=f"Database name to search articles. Supported values: {SUPPORTING_DB}")
    parser.add_argument("--output-csv", type=str, default="./articles.csv", help="Output CSV file for saving results")
    parser.add_argument("--keyword", type=str, nargs='+', default=None, help="Keyword(s) to filter articles")
    args = parser.parse_args()

    search_articles(args)