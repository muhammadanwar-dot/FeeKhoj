# Reflections on the Task

## 1. What part of the app depended most on the quality of the knowledge base?
The most knowledge-base-dependent part of the app was not the interface, and not even the LLM prompt. It was the mapping between the user’s problem and the charging rule. If the knowledge base misstates a statutory bucket, collapses distinct categories into one, or encodes the wrong ad valorem slab, the final answer becomes wrong even if the parser and interface work perfectly.

In this app, the knowledge base mattered in two especially important ways.

First, it had to preserve the difference between provisions that may look similar in ordinary language but trigger different charging consequences. A user may loosely say “I want an injunction regarding my property,” but the legal consequence changes if title is denied or if the suit is really one for declaration plus consequential relief. The app could not solve that problem through general language generation alone. It needed the knowledge base to represent those distinctions in a structured way.

Second, the ad valorem table had to be encoded carefully. That table is where legal text becomes computation. If the slab edges, base fees, or percentages are wrong, the app gives a mathematically precise but legally unsound output. That is a more dangerous failure than a merely clumsy explanation because the user may trust the number.

So the knowledge base was the backbone of the app. The LLM added convenience, but the database determined whether the app could be legally meaningful at all.

## 2. In what way did the LLM genuinely improve the app?
The LLM genuinely improved the app by translating unstructured user narratives into structured legal inputs. That is a real improvement because ordinary users usually do not think in terms such as “section 24(b) declaratory suit with consequential injunction concerning immovable property.” They describe facts. The LLM helps bridge that gap.

This improvement is not merely cosmetic. Without the extraction layer, the app would force the user into a lawyer-like form at the outset. The LLM allows the interface to be more public-facing by turning natural-language descriptions into a narrow schema: suit category, relevant value, and whether a title dispute is implicated.

The LLM also improved flexibility. It could handle a range of surface phrasings for recovery claims, declaratory claims, and injunction disputes without the user needing to learn the exact legal vocabulary built into the code.

## 3. In what way did the LLM create risks?
The principal risk was misclassification. The app’s legal consequences are narrow, but they are still sensitive to categorisation. If the model classifies a mixed-relief property dispute as a simple injunction, or treats a declaration-plus-possession case as a money dispute because the user mentions a sale price, the app may produce a clean number on the wrong legal basis.

A second risk was false confidence. LLM output often appears orderly and complete even when it is only partially grounded. In this app, that risk is especially acute because the model is being used at the threshold stage. A small interpretive mistake at the extraction stage propagates through the deterministic calculator and becomes an apparently authoritative fee estimate.

A third risk was ambiguity laundering. Users often give incomplete facts. An LLM can smooth over that incompleteness by inferring more certainty than the input justifies. In law, however, ambiguity should often be surfaced, not hidden. For example, the proper valuation may depend on whether the relief is tied to possession, title denial, or a stated relief value in the plaint. If the model fills those gaps too confidently, the tool becomes misleading.

## 4. What kinds of user queries did the app handle well, and what kinds did it handle badly?
The app handled well those queries that were both legally bounded and factually concrete. It worked best for:
- straightforward money recovery claims with a clear amount,
- declaration + injunction property disputes where the user states the property value,
- injunction disputes where the user clearly indicates that title is denied,
- other supported categories where the relevant monetary input is explicit.

It handled badly queries that were structurally messy or legally mixed. It was weak on:
- disputes involving multiple reliefs that may require more than one valuation lens,
- cases where the user does not know the market value of property,
- procedural questions rather than valuation questions,
- queries outside the supported statutory categories,
- narratives where the same facts could plausibly fit more than one charging provision.

The reason is straightforward: the app is strongest where the law is rule-like and the facts needed for computation are clearly stated. It is weakest where legal characterisation itself is contestable.

## 5. Did building this app change my view of what legal tech can and cannot do?
Yes. It reinforced a narrower but more serious view of legal tech. Before building the app, it is easy to imagine that legal-tech value comes mainly from conversational breadth. After building it, the more important insight is almost the opposite: legal tech is strongest when it is disciplined, bounded, and architecturally honest about where uncertainty lies.

This task made clear that legal tech can do useful public-facing work where the problem is structured enough to separate:
- fact intake,
- legal categorisation,
- and deterministic consequence.

It also made equally clear that legal tech is weak when it tries to collapse those layers into one smooth answer. Many legal problems are not just “information retrieval” problems. They involve contestable classification, incomplete facts, and consequences that depend on doctrinal distinctions the user may not even know are relevant.

So the exercise changed my view in a constructive way. I came away less impressed by broad chatbot-style legal tools and more persuaded by narrow systems that combine structured law with carefully constrained AI.

## 6. What would I change if this were to be used by real members of the public?
If this were intended for real public use, I would make five changes.

First, I would substantially expand the knowledge base and expose the statutory basis more transparently in the interface. A public tool should not only output a number; it should show the user the exact provision and logic path used.

Second, I would add a mandatory confirmation layer after extraction. The system should ask the user to confirm whether the dispute is indeed, for example, a money suit or a declaration-plus-injunction suit before calculating anything.

Third, I would build stronger ambiguity handling. Instead of forcing a single classification, the app should sometimes say that two statutory routes are plausible and explain what additional fact is needed to decide between them.

Fourth, I would create a more careful warning and escalation system. When the user’s problem involves mixed reliefs, unclear valuation, or unsupported categories, the app should stop and direct the user to legal assistance rather than pushing through a likely-wrong estimate.

Fifth, I would test the system against adversarial examples and real user language. In public deployment, robustness is not measured by whether the app works on clean examples, but by whether it fails safely on bad ones.

## 7. Final reflection
The most important lesson from the build was that the value of legal tech lies less in sounding intelligent and more in allocating intelligence correctly. In this app, the LLM was useful only because it was confined to translation and extraction. The actual legal consequence had to be anchored in a structured knowledge base and deterministic rules. That division of labour is, in my view, the central design lesson of the exercise.
