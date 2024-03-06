import sys
sys.path.append('MindSync')
import openai
import os
import arxiv
import functools

openai.api_key =  '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'

from agents import GeneralAgent
from paper_class import Paper
from paper_collection import PaperCollection
from paper_chat import PaperChat
from paper_source import PaperSource
from datetime import datetime

############################################################################################################

class ResearchAssistant(GeneralAgent):
    role_ = "You are a professional researcher, you are excellent at proposing the potential cross discipline study \
research topic between the area of study of AI Agent/multimodality/Large Language Model/Artificial General Intelligence/generative AI \
with regarding to the given research topic."

class ResearchTopicParser(GeneralAgent):
    role_ = '''You are a professional AI parser, and given a series of research topics, you always parse them into \
a python dict. For example: If you are given the following series of research topics:
1. title of this research topic: a concisive description with 1 sentence.
2. title of this research topic: a concisive description with 1 sentence.
...

Your output is a python dict as following:
{
  "title of this research topic": "a concisive description with 1 sentence.",
  "title of this research topic": "a concisive description with 1 sentence.",
  ...
}
'''

Potential_Research = '''
{research_interests}
'''

class Professor(object):
    def __init__(self,
                 first_name: str,
                 last_name: str,
                 research_interests: str):
        self.first_name = first_name
        self.last_name = last_name
        self.research_interests = research_interests

    def full_name(self):
        full_name = ' '.join([self.first_name, self.last_name])
        return full_name

    def full_name_convert(self):
        username_arxiv = f"{self.last_name.lower()}_{self.first_name[0].lower()}"
        return username_arxiv


class ResearchTopicComposer(object):
    def __init__(self,
                 research_interests: str,
                 papers: dict[str, Paper],
                 model: str = 'gpt-3.5-turbo',
                 num_retrieval: int = None,
                 score_threshold: float = 0.5):
        self.research_interests_ = research_interests

        chat = PaperChat(PaperSource(papers, openai.api_key))

        self.source_and_summarize_ = functools.partial(chat.source_and_summarize,
            num_retrieval=num_retrieval,
            score_threshold=score_threshold)

        self.researcher_: ResearchAssistant = ResearchAssistant(model)

    def get_raw_research_topics(self) -> dict[str,str]:
    
        prompt: str = f'''You will return a list of the three most promising potential areas of the proposed cross discipline research projects \
and topics with bullet points for the following research interests:

{self.research_interests_}

Please be professional, humble and polite. Please do not include people name. \
Just return the bullet points only, no explanations needed:

1. "a concisive title of the first research topic": a concisive description with 1 sentence for this topic.
2. "a concisive title of the second research topic": a concisive description with 1 sentence for this topic.
3. "a concisive title of the third research topic": a concisive description with 1 sentence for this topic.
...
'''

        potential_research_topic_text: str = self.researcher_.query(
            user_query=prompt,
            temperature=1, # the temp is 1, becaues the agent needs to be creative
        )

        research_topic_parser: ResearchTopicParser = ResearchTopicParser()
        parsed_topics: str = research_topic_parser.query(
            user_query=potential_research_topic_text,
            temperature=0,
        )
        return eval(parsed_topics)

    def _refine_research_topic(self, raw_potential_research_topic: str):
        '''
        Read the teacher's paper and combine with the interests to propose that somthing similar in teacher's research.

        Args:
            raw_potential_research_topic (str): The raw potential research topic from GPT.

        Returns:
            str: {raw_potential_research_topic}: {potential_topic}.
        '''

        sources: list = self.source_and_summarize_(query=raw_potential_research_topic)

        prof_related_research_works: str = '\n'.join([str(source[0].metadata) for source in sources])

        prompt = f'''As a world-class researcher, you are gonging to write to another professor.
You have a potential research topic {raw_potential_research_topic}. \
Please take a deep breath and propose the detailed methodologies that we could ellaborate based on the recent research works for another professor:

{prof_related_research_works}

Please return something like:
In your recent work "[put the tiltle of paper source here]", you proposed/mentioned the concept/method/conclusion of [XXX]. \
I posit that a more exhaustive investigation \
can be pursued in the direction of [put your potential research methodology/concept here]

NOTED:
Please provide a concisive, short paragraph.
Please be technically detailed description of what you believe to be the most promising research topic, along with relevant methodologies.
Please limit the explanation to one paragraph and maintain a humble tone.
Please DO NOT write "by Professor XXX".
Please write in the first person to another professor.
'''

        potential_topic: str =  self.researcher_.query(
            user_query=prompt,
            temperature=0.5,
            # temperature=1,
        )

        return f"{raw_potential_research_topic}: {potential_topic}"

    def get_research_topics(self, raw_potential_research_topics: dict[str, str] | None = None) -> str:
        '''get every topics in the begging with new topics'''
        if not raw_potential_research_topics:
            raw_potential_research_topics: dict = self.get_raw_research_topics()
        finetuned_potential_research_topics = []
        
        for index, topic in enumerate(raw_potential_research_topics):
            refined_topic = self._refine_research_topic(topic)
            finetuned_potential_research_topics.append(f"{index+1}. {refined_topic}")

        # return all finetuned_potential_research_topics in str 'topic1:xxxx \n topic2:xxxx'
        return '\n\n'.join(finetuned_potential_research_topics)



professor = Professor(
    first_name='Razvan',
    last_name='Marinescu',
    research_interests='''
Machine Learning
Generative ML Models
Bayesian inference
Applications of Machine Learning to Medicine
Molecular Biology
Molecular Dynamics
Entrepreneurship
'''
)

paper_collection = PaperCollection(
    openai_api_key=openai.api_key,
    chunk_size=2000,
)

# arXiv mode: get all your paper about query in arxiv
paper_collection.add_from_arxiv(
    search=arxiv.Search(
        id_list = ['2002.03419', '2107.09700'],
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending
    )
)

# Customized mode
# paper_collection.add_paper(Paper(
#     title='',
#     authors=[professor.full_name()],
#     summary="Avoid",
#     publish_date=datetime.strptime("2023", "%Y"),
#     url=''),)

research_topic_composer: ResearchTopicComposer = ResearchTopicComposer(
    research_interests=professor.research_interests,
    papers=paper_collection.papers,
    num_retrieval=5,
    score_threshold=0.5)

email_content = Potential_Research.format(
    research_interests=research_topic_composer.get_research_topics(),
)
print(email_content)
