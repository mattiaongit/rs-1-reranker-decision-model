"""Multiple-choice questions with the reranker: score every option against the question.

Two phrasings: the option alone, and the option with all choices listed in the query
(so the model can compare). Scores are also normalised across the options.

    uv run mcq.py
"""
from reranker import Reranker

QUESTIONS = [
    # easy: one clearly right answer
    {"question": "What is the capital of France?",
     "options": {"A": "Berlin", "B": "Madrid", "C": "Paris", "D": "Rome", "E": "Lisbon"}, "answer": "C"},
    {"question": "Which animal produces honey?",
     "options": {"A": "Ant", "B": "Bee", "C": "Butterfly", "D": "Wasp", "E": "Spider"}, "answer": "B"},
    {"question": "What do you use to cut paper?",
     "options": {"A": "Spoon", "B": "Scissors", "C": "Pillow", "D": "Hammer", "E": "Towel"}, "answer": "B"},
    # ambiguous: several defensible answers, the "most likely" one is a judgement call
    {"question": "Where would you most likely find a bat?",
     "options": {"A": "Cave", "B": "Baseball game", "C": "Attic", "D": "Zoo", "E": "Sporting goods store"}, "answer": "A?"},
    {"question": "What would you most likely use to open a stuck jar lid?",
     "options": {"A": "Rubber glove", "B": "Knife", "C": "Hot water", "D": "Towel", "E": "Hammer"}, "answer": "A/C?"},
    {"question": "Who is most likely to give you a ticket?",
     "options": {"A": "Cinema cashier", "B": "Police officer", "C": "Train conductor", "D": "Lottery vendor", "E": "Concert promoter"}, "answer": "?"},
]

INSTRUCTION = "Given a multiple-choice question, judge whether the Document is the correct answer to the Query."


def show(q, scores, label):
    total = sum(scores) or 1
    best = max(range(len(scores)), key=scores.__getitem__)
    print(f"  {label}")
    for i, (k, v) in enumerate(q["options"].items()):
        mark = " ◀" if i == best else ""
        print(f"    {k}) {v:<22} p(yes)={scores[i]:.3f}  share={scores[i] / total:5.1%}{mark}")


if __name__ == "__main__":
    rr = Reranker(batch_size=8)
    for q in QUESTIONS:
        opts = list(q["options"].values())
        print(f"\n{q['question']}   (expected: {q['answer']})")
        show(q, rr.score(INSTRUCTION, q["question"], opts), "option alone:")
        listed = q["question"] + " Choices: " + "; ".join(opts)
        show(q, rr.score(INSTRUCTION, listed, opts), "with all choices in the query:")
