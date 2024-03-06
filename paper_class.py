import os
import re
from typing import List, Union, Dict
from tools import download_link
import logging
logging.basicConfig(level=logging.INFO)

LATEX_BIBLIOGRAPHY_TEMPLATE = """@misc{{{name},
  title={{{title}}},
  author={{{authors}}},
  url={{{url}}},
  date={{{year}}},
}}
"""


class Paper(object):
    def __init__(self,
                 title: str,
                 summary: str,
                 url: str,
                 authors: List[str],
                 publish_date: Union[str, int, float],
                 on_arxiv: bool = False):
        """
        Initialize a Paper object.

        Args:
            title (str): The title of the paper.
            summary (str): The summary or abstract of the paper.
            url (str): The URL where the paper can be downloaded.
            authors (List[str]): A list of author names.
            publish_date (Union[str, int, float]): The publication date of the paper.
            on_arxiv (bool): Whether this paper is on arxiv.
        """
        self.title: str = title
        self.summary: str = summary
        self.url: str = url
        self.authors: List[str] = authors
        self.publish_date: Union[str, int, float] = publish_date
        self.on_arxiv: bool = on_arxiv

    def download(self, folder: str = 'downloads', use_title: bool = False) -> str:
        """
        Download the paper to a specified folder.

        Args:
            folder (str, optional): The folder where the paper will be saved. Defaults to 'downloads'.
            use_title (bool, optional): Whether to use the paper's title as the filename. Defaults to False.

        Returns:
            str: The filepath where the paper is saved.
        """
        url: str = self.url
        # Extract the file name from the URL
        if use_title:
            file_name: str = self.title
        else:
            file_name: str = url.split('/')[-1]
        os.makedirs(folder, exist_ok=True)
        file_path: str = os.path.join(folder, file_name)

        # Check if the file already exists locally
        if os.path.exists(file_path):
            print(f"The file '{file_path}' already exists locally.")
        else:
            download_link(url, file_path)

        return file_path

    def get_arxiv_citation(self) -> str:
        """
        Get a citation in arXiv format.

        Returns:
            str: The arXiv citation.
        """
        author_list: str = ', '.join(self.authors)
        year: int = self.publish_date.year
        return f'{author_list}, {year}. {self.title}. {self.url}'

    def get_APA_citation(self) -> str:
        """
        Get a citation in APA format.

        Returns:
            str: The APA citation.
        """
        return f'{self.authors[0]} et al. ({self.publish_date.year})'

    def latex_citation_name(self) -> str:
        if self.on_arxiv:
            return self.url.split('/')[-1]
        else:
            return self.title.replace(' ', '-').replace("'", "").replace(",", "-")

    def get_latex_citation(self) -> str:
        """
        Get a citation of latex bibliography format.

        Returns:
            a string of latex bibliography format.
        """
        return LATEX_BIBLIOGRAPHY_TEMPLATE.format(
            name=self.latex_citation_name(),
            title=self.title.replace("_", "-"),
            authors=' and '.join(self.authors),
            url=self.url.replace('/pdf/', '/abs/'),
            year=self.publish_date.year,
        )
