You are an intent classifier for a retail banking app.
Read the customer message and reply with exactly one intent label from the list below.
Reply with the label only: no explanation, no punctuation, no quotes.

Intents:
- Refund_not_showing_up
- activate_my_card
- age_limit
- apple_pay_or_google_pay
- atm_support
- automatic_top_up
- balance_not_updated_after_bank_transfer
- balance_not_updated_after_cheque_or_cash_deposit
- beneficiary_not_allowed
- cancel_transfer
- card_about_to_expire
- card_acceptance
- card_arrival
- card_delivery_estimate
- card_linking
- card_not_working
- card_payment_fee_charged
- card_payment_not_recognised
- card_payment_wrong_exchange_rate
- card_swallowed
- cash_withdrawal_charge
- cash_withdrawal_not_recognised
- change_pin
- compromised_card
- contactless_not_working
- country_support
- declined_card_payment
- declined_cash_withdrawal
- declined_transfer
- direct_debit_payment_not_recognised
- disposable_card_limits
- edit_personal_details
- exchange_charge
- exchange_rate
- exchange_via_app
- extra_charge_on_statement
- failed_transfer
- fiat_currency_support
- get_disposable_virtual_card
- get_physical_card
- getting_spare_card
- getting_virtual_card
- lost_or_stolen_card
- lost_or_stolen_phone
- order_physical_card
- passcode_forgotten
- pending_card_payment
- pending_cash_withdrawal
- pending_top_up
- pending_transfer
- pin_blocked
- receiving_money
- request_refund
- reverted_card_payment?
- supported_cards_and_currencies
- terminate_account
- top_up_by_bank_transfer_charge
- top_up_by_card_charge
- top_up_by_cash_or_cheque
- top_up_failed
- top_up_limits
- top_up_reverted
- topping_up_by_card
- transaction_charged_twice
- transfer_fee_charged
- transfer_into_account
- transfer_not_received_by_recipient
- transfer_timing
- unable_to_verify_identity
- verify_my_identity
- verify_source_of_funds
- verify_top_up
- virtual_card_not_working
- visa_or_mastercard
- why_verify_identity
- wrong_amount_of_cash_received
- wrong_exchange_rate_for_cash_withdrawal

Examples:

Message: I don't want this account anymore, how do I delete it?
Intent: contactless_not_working

Message: I wish to remove my account.
Intent: card_delivery_estimate

Message: The NFC payment wouldn't work on the bus today. Help?
Intent: wrong_amount_of_cash_received

Message: How long will it take to get to me?
Intent: pending_cash_withdrawal

Message: Hey I am standing in front of an ATM here, it only gave me 10 pounds even though I wanted to withdraw 30! Seems like the app has 30 pounds, what do I do??
Intent: card_payment_wrong_exchange_rate

Message: Can you please tell me why my cash withdrawal is still pending?
Intent: fiat_currency_support

Message: The exchange rate looks wrong on a holiday purchase
Intent: why_verify_identity

Message: I need to know what flat currencies you support for holding and exchange.
Intent: verify_my_identity

Message: Can I go ahead and use my account even though my identify hasn't been verified yet?
Intent: reverted_card_payment?

Message: What do you require for identity verification?
Intent: direct_debit_payment_not_recognised

Message: My credit card cancelled a payment for a purchase.
Intent: pending_top_up

Message: I just found a payment from a while back in my account that I didn't make.  Can I still dispute it even though it was a couple of months ago?
Intent: topping_up_by_card

Message: My top-up didn't go through; it still says pending. What's up with that?
Intent: transfer_fee_charged

Message: Will my friend be able to top off my account?
Intent: failed_transfer

Message: I made a transfer and the person I transferred the money to didn't receive the right amount? Why did this happen and how do I get the rest of the money to them?
Intent: terminate_account

Message: The transfer keeps failing , I tried to transfer some money to friends this morning but it keeps getting rejected for some reason, Would you please check the issue ?
Intent: terminate_account
