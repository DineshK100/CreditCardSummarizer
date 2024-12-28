from groq import Groq
import pandas as pd

client = Groq()


def compare_summaries(pairwise_file, system_prompt, output_file):
    # Load pairwise comparison data
    pairs = pd.read_csv(pairwise_file)

    # Prepare results
    results = []

    for _, row in pairs.iterrows():
        summary_1 = row["Summary 1"]
        summary_2 = row["Summary 2"]

        # Send summaries to the LLM-as-Judge
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Summary 1: {summary_1}\n\nSummary 2: {summary_2}\n\nWhich summary is better? Please respond with 'Summary 1' or 'Summary 2'.",
                },
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )

        # Collect the result
        decision = ""
        for chunk in completion:
            decision += chunk.choices[0].delta.content or ""

        results.append(
            {
                "Summary 1": summary_1,
                "Summary 2": summary_2,
                "Decision": decision.strip(),
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Comparison results saved to {output_file}")


if __name__ == "__main__":

    # Provide two summaries to compare
    summary_1 = "Text goes here"

    summary_2 = "Text goes here"

    system_prompt = """Here are good quality summaries:\n\nFirst Command Bank Visa Cardholder Agreement (Platinum/Classic) The person(s) ("Cardholder," whether one or more) who signed and returned the Application for a Visa ("Application") has requested First Command Bank ("Bank") to extend to Cardholder open-end credit. anyone else using the Card unless the use of such Card is by a person other than the Cardholder. Bank will inform Cardholder from time to time of the maximum amount of debt ("Credit Limit") that may be outstanding in the Account at any time. Cardholder agrees not to use the Card in any manner that would cause the outstanding balance to exceed the Credit Limit. Bank may designate that only a portion of Cardholder\'s Credit Limit is available for Cash Advances. and providing Cardholder information about products and services. As of the end of each monthly billing cycle, Cardholder will be furnished a periodic statement showing, among other things, (a) the amount owed ("Previous Balance") at the beginning of the billing cycle. If Cardholder is composed of more than one person, only one periodic statement will be provided. Charge, Bank begins to charge the Finance Charge on all amounts Cardholder owes Bank (except "New Purchases") from the first day of the billing cycle. A New Purchase is one that appears on the periodic statement for the first time. Bank calculates the Finance charge on the Account by applying the "Periodic Rate" (defined below) to the "Average Daily Balance" Bank adds 3% to the Prime Rate to determine Platinum APR (Periodic Rate currently .005208). For Credit Purchases and Cash Advances which occur on a Platinum Account, the APR varies with changes to the prime rate. All payments received on or before 5 o\'clock p.m. (Fort Worth, Texas time) on Bank\'s business day at the address indicated on the periodic statement will be credited to the Account. Convenience Check, the instructions which Bank provides when the Convenience Checks are issued must be followed. Cardholder agrees to hold Bank harmless and indemnify Bank for any losses, expenses and costs, including attorney\'s fees incurred by Bank. Bank will use its best efforts to stop payment, but will incur no liability if it is unable to. then owed to Bank by Cardholder immediately due and payable, without prior notice or demand of any kind, except as may be required by applicable law. Bank may increase the Annual Percentage Rate up to 18%, which is the Default Rate under the Table of Charges. Cardholder agrees to pay all amounts actually incurred by Bank as court costs and attorneys\' fees. not complete a transaction to or from the Account on time or in the correct amount according to this Agreement, Bank may be liable for Cardhold- er\'s losses or damages. Bank will not be liable if, through no fault of Bank\'s, the available credit is insufficient for the transaction or is unavailable for withdrawal. Cardholder authorizes Bank to share information about Cardholder\'s payment history with other persons or companies when permitted by law. Bank will not be responsible for merchandise or services purchased by Cardholder with the Card or Convenience Check unless required by law. Any refund, adjustment or credit allowed by a Seller shall not be cash, but rather be by a credit advice to Bank which shall be shown as a credit on the periodic statement. Bank is subject to the requirement of the USA Patriot Act. Bank may obtain at any time Cardholder\'s credit reports for any legitimate purpose associated with the Account or the application or request for an Account. Ohio anti-discrimination laws require creditors to make credit equally available to all creditworthy customers. Married Wisconsin Applicants: No provisions of any marital property agreement, unilateral statement, or court order applying to marital property will adversely affect a creditor\'s interests unless prior to the time credit is granted. on the goods or services. You have this protection only when the purchase price was more than $50 and the purchase was made in your home state. (If we own or operate the merchant, or if we mailed the advertisement for the property or services, all pur- chases are covered regardless of amount or location of purchase.)\n\n\nThis Agreement covers this VISA Platinum Credit Card issued by iQ Credit Union. You promise to pay us in United States dollars, by cash, check or money order. You agree to pay advances requested by any co-applicant the same as if you asked for the loan and it was paid to you. You may earn a 1.00% cash rebate on your Account if you meet the following requirements. Cash advance transactions, including convenience checks and balance transfers, do not qualify for the cash rebate. The cash rebate will be calculated each quarter based on the average daily balance in your Account. pay each month not less than the minimum payment on or before the scheduled due date. Minimum payment will be three percent (3.0%) of your outstanding balance or $25.00, whichever is greater, plus the greater of any amount past due or any amount in excess of your credit line. The total outstanding balance of purchases and cash advances in the Account on the closing date of a billing cycle will be shown on the Periodic Statement for that billing cycle as the �New Balance. A FINANCE CHARGE will be imposed on cash advances from the date each cash advance is made. There is no time period within which to pay to avoid a periodic FINANCES CHARGE oncash advances. In addition, a cash advance fee (FINANCE CHARge) equal to 2% of the cash advance (or $10.00, whichever is greater) will be applied to each cash advances. There is no annual fee. The fee for a cash advance is the greater of 2.00% of the cash advance amount or $5.00. The Daily Periodic Rate for cash advances (including balance transfers) is .046547%, with a corresponding Annual PERCENTAGE RATE of 16.99%. You pledge all of your present and future shares and any earnings thereon as security for obligations under your Account. Purchases and Cash Advances made in foreign countries will be billed to you in U.S. dollars. We may issue you a Personal Identication Number for using your Card to obtain cash advances at automatic teller machines. anyone not authorized to sign on your Accounts. To keep your Account secure, please do not write your PIN on your Card or keep it in the same place as your Card. Default. You will be in default under this Agreement if any of the following occur: (a) Any minimum payment is not made when due; (b) You become insolvent, bankrupt, or you die; (c) You violate any part of this Agreement, or any other agreement. Platinum Credit Card transactions under the Fair Credit Billing Act. If you think your bill is wrong, or if you need more information about a transaction on your bill, write to us at the address listed above. We must hear from you no later than 60 days after we sent you the .rst bill on which the error or problem appeared. purchased with a credit card, and you have tried in good faith to correct the problem with the merchant, you may have the right not to pay the remaining amount due on the property or services. There are two limitations on this right: i. You must have made the purchase in your home state or within 100 miles of your current mailing address; and ii. The purchase price must have been more than $50.00.\n\nThe words "you", "your", and/or "Cardholder(s)" refer to the person whose name is embossed in the face of the MasterCard(r) Platinum, MasterCard Platinum Cash and Master card Platinum Rewards credit card. CFE reserves the right to amend this Agreement at any time and for any reason. You agree not to use your Card to conduct any illegal transaction per applicable federal, state, or local law. You agree to repay according to the terms of this Agreement all transactions you initiate by use of your card, whether deemed legal or illegal. CFE Federal Credit Union advises all check card and credit cardholders to please exercise discretion when using an Automated Teller Machine (ATM) You agree to pay all Finance Charges and other charges added to your Account under the terms of this Disclosure and Agreement and any other applicable Agreement you have made with CFE. You undertake to safeguard the Card against damage, loss, theft or misuse and to maintain it at all times in a safe place. The Credit Card Act restricts CFE\'s ability to provide young members with credit cards who are under the age of 21. Account activity for the prior period ("cycle"), including purchases, cash advances, fees, and payments. Payment: You agree to pay at least the minimum required payment amount shown on your statement by the date shown on the statement or no later than 25 days from the statement closing date. Payment Allocation: Payments in excess of the Minimum Payment Due will be applied to the balance with the highest annual percentage rate. any credit application or credit update; 6) in the event of your death; 7) if you incur charges for Purchases and Cash Advances which exceed the maximum authorized credit limit; and/or 8) if the Issuer, in its sole discretion, believes your ability to repay what you owe may be substantially reduced due to an event either within or beyond your control. Failure by Issuer to assert any rights hereunder shall not waive such rights. The Card(s) issued to you for this Account remain the property of the Issuer. CFE reserves the right to reinvestigate and reevaluate any information you provided on your credit application at anytime. You agree that we may release information to others, such as credit bureaus, regarding the status and history of your account. You agree to submit to the jurisdiction of the State of Florida. The exclusive venue for any action or dispute shall be in the courts of Orange County, Florida. You agree that your Account shall be subject to all applicable rules and regulations of MasterCard International, as applicable, as well as all applicable laws. We take the beginning outstanding balance of purchases each day, add any new purchases, and subtract any payments and/or credits. Then, we add all the daily balances of purchases together and divide the total by the number of days in the billing cycle. This gives us the average daily balance.of purchases. Balance transfers must be transferred from any financial institution other than CFE. of $20 or 3% of the outstanding balance not to exceed $25. Penalty APR: Each time your account becomes 60 days past due, the Penalty APR will apply to all account balances. Returned Check or Rejected Payment: If you pay us electronically or by check and your financial institution does not honor the payment, a fee of $25 will apply. Conversion: For MasterCard, if you effect a transaction with your MasterCard in a currency other than U.S. dollars, MasterCard International Incorporated will convert the charge. Protected Balances Due to Rate Increases: Protected balances are the amount owed for a category of transactions to which an increased annual percentage rate cannot be applied. No annual fee applies to this credit card. the end of the day on which credit availability is terminated or suspended. CFE will establish an amortization period required by law for the balance on the account. The right to reject a change in terms does not apply if CFE has not received a member\'s required minimum periodic payment within 60 days after the due date.\n\n\nKeep that in mind. \n\nNow I will give you two summaries and I want you to pick the best one based on the baseline summaries I gave you. """

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
            {
                "role": "user",
                "content": f"Summary 1: {summary_1}\n\nSummary 2: {summary_2}\n\nWhich summary is better? Please respond with 'Summary 1' or 'Summary 2'.",
            },
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Print the model's response
    print("\nModel's decision:")
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")
