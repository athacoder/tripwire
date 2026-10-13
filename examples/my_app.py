"""A stand-in for your application: any function from a case's input to an answer.

Tripwire calls it once per case with the case's `input` and scores what it returns. This
one guesses a banking intent from a few keywords, which is enough to have something to
gate. Return a dict with an "output" key to attach more, such as a trace id.
"""

KEYWORDS = {
    "card_arrival": ("arrive", "arrived", "delivery"),
    "lost_or_stolen_card": ("lost", "stolen"),
    "exchange_rate": ("exchange rate", "rate"),
    "top_up_failed": ("top up", "top-up", "topping"),
    "change_pin": ("pin",),
}


def answer(case: dict) -> str:
    text = case["text"].lower()
    return next((intent for intent, words in KEYWORDS.items() if any(w in text for w in words)), "")
