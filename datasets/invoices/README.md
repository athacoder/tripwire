# Synthetic invoices

300 short text invoices for structured extraction. `scripts/make_invoices.py` generates
them with seed 11.

Each invoice is rendered from random parameters (vendor, invoice number, date, currency,
line items, tax rate) in one of four layouts and one of four date styles. The expected
record is built from the same parameters:

```json
{"vendor": "...", "invoice_number": "...", "date": "YYYY-MM-DD", "currency": "USD", "total": 1234.5}
```

Because the answers follow from the generator, they are correct by construction: no model
and no annotator was involved. The invoices exist nowhere else, so the task cannot be
answered from memory.

Cases are tagged by layout, currency and number of line items. The suite scores them with
`field_f1`, which compares each field after normalising text and parsing numbers.

## Limits

- **Saturated for now.** `gemma3:4b` scores 0.999 on it (the few misses are wrong
  dates), and `tripwire report` warns that a score this high cannot
  tell configurations apart. The suite exercises the continuous-metric path; it needs
  harder layouts before it can gate anything.
- The layouts are simple and clean. Real invoices are messier; a high score here says
  little about scanned documents.
- Vendors and item names come from short fixed lists.
