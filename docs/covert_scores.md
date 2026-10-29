# Covert and overt scores

Three numbers are added to the inputs right before `eac_protect_covert`, as in the existing pipeline:

| Score | How it is calculated |
|---|---|
| `overt_score` | number of YES answers among the 9 overt questions (0 to 9) |
| `covert_score` | number of YES answers among the 9 covert questions (0 to 9) |
| `behavioral_bias_ratio` | `overt_score / (covert_score + 1)` |

## Counting rules

- An answer counts only if, once trimmed and upper-cased, it equals `YES`.
- Blank answers add 0.
- A question missing from the data counts as NO.
- Question texts must match exactly after header cleaning: curly apostrophes become straight ones. The typo "point it our" is part of the real column name.

## Overt questions (9)

1. Would you describe yourself as direct?
2. Would you describe yourself as desiring for things to be fair or just?
3. Do you have a pattern of saying things impulsively that you would later regret and feel the desire to fix things or smooth them over?
4. Even if you feel like you're trying to help, do people blame you for being overly direct, controlling or quick to anger?
5. Are you quick to correct or step in verbally in a way that often creates conflict or hurt feelings with others?
6. Do you ask direct or overt questions?
7. When a boundary has been crossed do you find yourself quick to point it our or stand up for yourself?
8. Do you get upset if something or someone attempts to derail your idea, plan or goal?
9. Do you ideally prefer control over a situation but expect things to go wrong?

## Covert questions (9)

1. Do you tend to be a people-pleaser or say what someone needs to hear rather than what is actually on your mind?
2. Do you tend to keep things to yourself because you don't want to upset others or rock the boat?
3. Is it challenging for you to voice your wants, needs or concerns?
4. Do you struggle with expressing boundaries or expectations?
5. Do you often people-please and not say anything until you can't take it anymore, and then say everything that you've been holding in?
6. Do you go through prolonged periods of keeping things in and staying quiet followed by needing to stand up for yourself or what feels fair?
7. Do you find it challenging to trust other people's judgment or ability to follow through on plans or a commitment?
8. Do you replay fights in your head or imagine what you should have said instead?
9. If you experience apathy and withdraw, do you tend to isolate and willingly self-sabotage?

## Average scores by label (from the EDA)

| Protect label | Overt | Covert |
|---|---|---|
| Anger | 6.3 | 5.5 |
| Covert Anger | 5.2 | 5.4 |
| Anxiety | 4.8 | 6.1 |

| Escalating label | Overt | Covert |
|---|---|---|
| Anxiety | 5.7 | 5.1 |
| Anger | 4.9 | 5.9 |
| Frustration | 3.6 | 6.4 |

The scores separate Anger from Anxiety well, but separate Covert Anger from plain Anger only weakly.

## Notes on the existing pipeline

- In the existing pipeline these scores are only computed on the Protect anger path. Any Escalating model that uses them must compute them for every respondent.
- On the covert path, the existing pipeline returns `Covert Anger` and does not check apathy. This flow adds `Covert Anger-Apathy`.
