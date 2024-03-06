from paper_class import Paper
from datetime import datetime


def get_test_papers():
    papers = {
        "paper 1": Paper(
            title="paper 1",
            summary="",
            authors=["Razvan V Marinescu", "Sungmin Hong"],
            url="https://arxiv.org/pdf/2107.09700",
            publish_date=datetime.strptime("2021", "%Y"),
            # 3d-stylegan: A style-based generative adversarial network for generative modeling of three-dimensional medical images
        ),
        "paper 2": Paper(
            title="paper 2",
            summary="",
            authors=["Razvan V Marinescu", "Neil P Oxtoby"],
            url="https://arxiv.org/pdf/2002.03419",
            publish_date=datetime.strptime("2021", "%Y"),
            # The Alzheimer's disease prediction of longitudinal evolution (TADPOLE) challenge: Results after 1 year follow-up
        ),
    }

    return papers
