"""Generate a synthetic invoice-extraction dataset whose answers are known by construction.

Every invoice is rendered from random parameters, and the expected record is built from the
same parameters, so no model and no annotator is involved in the gold labels. The task
cannot be answered from memory: none of these invoices exists anywhere else.

    python scripts/make_invoices.py
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from tripwire import datasets
from tripwire.models import Case

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "invoices" / "extract.jsonl"
SEED, COUNT = 11, 300

VENDORS = [
    "Northwind Traders", "Halden & Voss GmbH", "Blue Kettle Supply Co.", "Orchard Lane Studio",
    "Kestrel Freight Ltd", "Marlow Office Systems", "Tindall Instruments", "Sable Ridge Catering",
    "Quarry Street Print", "Ferro Tooling SA", "Lumen Works", "Pemberton Analytics",
]  # fmt: skip
ITEMS = [
    "Consulting hours", "Laser toner", "Site inspection", "Annual licence", "Freight handling",
    "Calibration service", "Printed brochures", "Standing desk", "Cloud storage", "Workshop day",
]  # fmt: skip
CURRENCIES = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
# The same date written four ways; the expected answer is always ISO.
DATE_STYLES = ["%d %B %Y", "%Y-%m-%d", "%d.%m.%Y", "%b %d, %Y"]

LAYOUTS = [
    "{vendor}\nINVOICE {number}\nDate: {date}\n\n{lines}\n\nSubtotal: {subtotal}\n"
    "Tax ({tax}%): {tax_amount}\nTOTAL DUE: {total}\n",
    "Invoice No. {number}                Issued {date}\nFrom: {vendor}\n\n"
    "Description | Qty | Unit | Amount\n{lines}\n\nNet {subtotal} / VAT {tax}% {tax_amount}\n"
    "Amount payable: {total}\n",
    "Thank you for your business.\n\nBilled by {vendor} on {date}.\nReference: {number}\n\n"
    "{lines}\n\nSum before tax {subtotal}; tax at {tax}% is {tax_amount}.\n"
    "Please pay {total} within 30 days.\n",
    "*** {vendor} ***\n{date}\n\n{lines}\n\nsubtotal {subtotal}\ntax {tax}% {tax_amount}\n"
    "total {total}\n\ninvoice ref {number}\n",
]


def money(amount: float, symbol: str) -> str:
    return f"{symbol}{amount:,.2f}"


def make(rng: random.Random, index: int) -> Case:
    layout = index % len(LAYOUTS)
    code = rng.choice(list(CURRENCIES))
    symbol = CURRENCIES[code]
    day = date(2024, 1, 1) + timedelta(days=rng.randrange(730))
    number = f"{rng.choice(['INV', 'BK', 'R', 'F'])}-{rng.randrange(1000, 99999)}"
    rows = [
        (rng.choice(ITEMS), rng.randint(1, 12), round(rng.uniform(4, 950), 2))
        for _ in range(rng.randint(1, 4))
    ]
    subtotal = round(sum(qty * price for _, qty, price in rows), 2)
    tax = rng.choice([0, 5, 8, 18, 20])
    tax_amount = round(subtotal * tax / 100, 2)
    total = round(subtotal + tax_amount, 2)
    vendor = rng.choice(VENDORS)
    text = LAYOUTS[layout].format(
        vendor=vendor,
        number=number,
        date=day.strftime(DATE_STYLES[rng.randrange(len(DATE_STYLES))]),
        lines="\n".join(
            f"{name} | {qty} | {money(price, symbol)} | {money(qty * price, symbol)}"
            for name, qty, price in rows
        ),
        subtotal=money(subtotal, symbol),
        tax=tax,
        tax_amount=money(tax_amount, symbol),
        total=money(total, symbol),
    )
    expected = {
        "vendor": vendor,
        "invoice_number": number,
        "date": day.isoformat(),
        "currency": code,
        "total": total,
    }
    tags = [f"layout:{layout}", f"currency:{code}", f"items:{len(rows)}"]
    return Case(input={"text": text}, expected=expected, tags=tags, source="template")


def main() -> None:
    rng = random.Random(SEED)
    cases = [make(rng, i) for i in range(COUNT)]
    datasets.save(OUT, cases)
    print(f"{len(cases)} invoices, version {datasets.version(cases)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
