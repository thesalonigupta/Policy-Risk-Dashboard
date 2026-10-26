# Labeling guide for 10-K risk sentences

Label each sentence 1 if it is substantively about the topic named in the `topic` column, otherwise 0. Judge the sentence
on its own, as written. You do not know (and should not guess) what any scoring method said about it.

**AI** — The sentence discusses artificial intelligence, machine learning, generative AI, large language models, or
automated/algorithmic decision models as a source of risk, change, or regulation. A passing mention counts if AI is
named as one of the risk drivers ("risks from cyber threats, AI and quantum computing" = 1). Quantitative trading or
risk models with no AI/ML framing = 0. Generic "new technologies" with no AI reference = 0.

**Regulatory** — The sentence discusses regulation, supervision, enforcement, compliance obligations, capital or
liquidity requirements, fines or penalties as a risk or constraint on the firm. Headings and cross-references that only
name a regulatory risk factor = 1 if they state the topic. Litigation or legal risk with no regulator/rules angle = 0.
A mere mention of "regulatory" inside an unrelated list (fund administration, custody) = 0.

**Political** — The sentence discusses government or political developments as a risk: elections, changes of
administration, government policy beyond financial regulation (tax, trade, fiscal), tariffs, trade conflict, sanctions,
geopolitical conflict or war, government shutdowns, political polarization. Financial regulation on its own is
Regulatory, not Political. Market stress with no political cause = 0.

Naming a topic as one driver of risk counts for AI and Political even in a list ("political or geopolitical events" = 1). For
Regulatory, a regulatory action named only as one possible consequence of an unrelated event (cyber incident, fraud) = 0.

When a sentence is truncated or a fragment, label what it says. When unsure, label 0 and note it in `notes`.
