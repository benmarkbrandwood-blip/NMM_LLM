# Nine Men's Morris human move policy evaluation report

## Overview
This report analyzes the held-out validation evaluation for a human move prediction model trained on Nine Men's Morris positions from human-played games. The validation run evaluated 384,837 samples and 785,550 move events, using a model trained with a count-weighted cross-entropy objective on a v2 dataset split with 79 input features and no temperature adjustment beyond `temperature_star = 1.0`.[cite:94]

Overall, the model is clearly learning human move preferences rather than merely guessing legal moves. On the full validation set it reaches event NLL 1.5816, Brier 0.6155, top-1 45.53%, top-3 79.49%, top-5 88.99%, and ECE 0.1750, versus a uniform baseline at event NLL 2.3474, Brier 0.8084, top-1 11.73%, top-3 31.24%, top-5 48.43%, and ECE 0.0200.[cite:94] The model therefore achieves much stronger ranking accuracy than uniform, but its non-zero ECE shows that confidence calibration is still imperfect.[cite:83][cite:88]

The core metrics have different roles. Event NLL measures how much probability mass the model assigns to the actual human move, so lower is better; Brier score measures squared probability error and is also lower-is-better; top-1/top-3/top-5 measure whether the played human move is within the model’s top k predictions, so higher is better; and ECE measures the mismatch between predicted confidence and empirical accuracy, so lower is better.[cite:79][cite:81][cite:83][cite:88]

## Metric interpretation

### NLL
Negative log-likelihood is the sharpest measure of whether the model puts probability on the exact move humans actually played. A lower NLL means the model not only ranks the right move reasonably high, but also gives it a strong enough probability mass rather than spreading probability too diffusely across many alternatives.[cite:79][cite:82]

### Brier score
The Brier score is a proper scoring rule based on squared error between the predicted probabilities and the actual outcome. In this setting, lower Brier indicates a better overall probability distribution, reflecting a mix of discrimination and calibration rather than only top-1 correctness.[cite:79][cite:81][cite:88]

### Top-k accuracy
Top-1 is the exact move match rate; top-3 and top-5 tell whether the real human move appears among the model’s short list of best guesses. For a move policy used to imitate human choice or support downstream search, strong top-3 and top-5 numbers often mean the model is capturing a meaningful candidate set even when the top prediction is not exact.[cite:90]

### ECE
Expected Calibration Error compares mean confidence with empirical accuracy across confidence bins. A low ECE indicates that the model’s probabilities are trustworthy as probabilities, whereas a larger ECE means the model may still rank moves well but be overconfident or underconfident about them.[cite:83][cite:85][cite:92]

## Overall performance
The overall model numbers indicate a strong shortlist predictor and a reasonable exact-move predictor on held-out data. A top-1 of 45.53% means that nearly half the time the most probable move matches the human move, while top-3 at 79.49% and top-5 at 88.99% indicate that the true human move is usually inside the model’s small candidate set.[cite:94]

Relative to the uniform baseline, the improvement is substantial across all ranking metrics. The model improves top-1 from 11.73% to 45.53%, top-3 from 31.24% to 79.49%, and top-5 from 48.43% to 88.99%, while also reducing event NLL from 2.3474 to 1.5816 and Brier from 0.8084 to 0.6155.[cite:94] This is strong evidence that the network has learned meaningful human move structure rather than just the legal move count distribution.[cite:94]

At the same time, calibration is only moderate. The model ECE of 0.1750 is much larger than the uniform baseline’s 0.0200, which is a reminder that a highly informative model can still be miscalibrated, especially when its predicted distribution is sharp rather than flat.[cite:94][cite:83][cite:92]

## Elo-band slices
The Elo-band slice shows that the model performs fairly consistently across lower, middle, and upper bands, with only modest variation. Lower-band positions score event NLL 1.6135, Brier 0.6201, top-1 44.78%, top-3 78.29%, top-5 87.99%, and ECE 0.1752; middle-band positions score NLL 1.5669, Brier 0.6090, top-1 46.13%, top-3 79.61%, top-5 89.30%, and ECE 0.1769; upper-band positions score NLL 1.5832, Brier 0.6190, top-1 45.32%, top-3 79.75%, top-5 89.05%, and ECE 0.1736.[cite:94]

The middle band is the easiest slice by NLL, Brier, and top-1. That suggests the model may be learning the most statistically regular or most densely represented human behaviour in this band, while lower-band play may be noisier and upper-band play may be slightly more diverse or tactically subtle.[cite:94]

The upper band still has the best top-3 of 79.75% and a slightly better ECE of 0.1736 than the other bands. That combination suggests the model is good at keeping expert human moves within its shortlist, even if the exact top-1 choice remains only slightly below the middle-band result.[cite:94]

## Phase slices
The phase split is one of the most informative slices. In the placing phase, the model has event NLL 1.8827, Brier 0.6444, top-1 39.34%, top-3 73.81%, top-5 84.03%, and ECE 0.2013; in the moving phase, it improves to NLL 1.3416, Brier 0.6067, top-1 50.48%, top-3 84.02%, top-5 92.94%, and ECE 0.1714.[cite:94]

This is a large gap and indicates that move-phase decisions are much easier for the model to imitate than placement-phase decisions. A plausible interpretation is that moving positions contain stronger local tactical and structural signals in the 79-dimensional feature representation, whereas placement choices may depend more on longer-horizon plans, opening preferences, or stylistic variation that are harder to infer from immediate features alone.[cite:94]

Calibration is also clearly worse in placing positions, with ECE 0.2013 versus 0.1714 in moving positions. So the opening/placement stage is not only harder to predict, but also a stage where the model’s confidence should be treated more cautiously.[cite:94][cite:83]

## Legal-move-count slices
The legal-move-count slices show a strong dependence on branching factor. For 2–5 legal moves, the model scores NLL 0.9004, Brier 0.4734, top-1 61.40%, top-3 95.67%, top-5 100.00%, and ECE 0.2113; for 6–10 legal moves, NLL 1.2984, Brier 0.5924, top-1 51.33%, top-3 85.30%, top-5 95.25%, and ECE 0.2173; for 11–20 legal moves, NLL 1.9605, Brier 0.7277, top-1 37.66%, top-3 64.71%, top-5 78.13%, and ECE 0.1990; and for 21+ legal moves, NLL 2.0888, Brier 0.8476, top-1 33.95%, top-3 74.74%, top-5 83.09%, and ECE 0.1817.[cite:94]

This pattern is exactly what would be expected from a policy model: the more legal moves there are, the harder exact prediction becomes. NLL and Brier worsen steadily from low-branching to high-branching positions, while top-1 drops from 61.40% to 33.95% as the decision space broadens.[cite:94]

Two details are interesting. First, the 21+ slice has worse NLL/Brier/top-1 than 11–20, but better top-3 and top-5, which suggests the model often still knows the broad region of plausible human choices even when the exact best-ranked move is wrong.[cite:94] Second, ECE is actually highest in the easiest low-branching slices, implying the model may become overconfident when the move set is small, even though its shortlist accuracy there is excellent.[cite:94][cite:83]

## Malom transition slices
The transition slice appears to label whether a move preserves, improves, or worsens the status of the position with respect to the Malom outcome labelling. These slices are especially useful because they separate routine play from blunder-like or strategically costly transitions.[cite:94]

The easiest transition category is `all_losing`, with NLL 1.3222, Brier 0.6288, top-1 51.52%, top-3 83.83%, top-5 92.74%, and ECE 0.1687.[cite:94] This suggests that when every continuation is losing, human choices are relatively predictable, perhaps because only a few defensive or delay-maximising moves remain plausible.[cite:94]

`draw_preserved` is the largest slice and behaves close to the global average, with NLL 1.5721, Brier 0.6696, top-1 46.29%, top-3 81.30%, top-5 89.83%, and ECE 0.1824.[cite:94] `win_preserved` is slightly harder, with NLL 1.6295, Brier 0.7032, top-1 45.10%, top-3 74.61%, top-5 86.47%, and ECE 0.1887, which may reflect a broader range of acceptable winning continuations.[cite:94]

The hardest labelled transitions are the degrading ones. `win_to_draw` yields NLL 2.0367, Brier 0.8547, top-1 26.69%, top-3 61.76%, top-5 79.03%, and ECE 0.1325; `draw_to_loss` yields NLL 2.2218, Brier 0.9125, top-1 21.43%, top-3 56.27%, top-5 73.82%, and ECE 0.1063; and `win_to_loss` is the most difficult, with NLL 2.5412, Brier 0.9685, top-1 15.29%, top-3 46.05%, top-5 67.82%, and ECE 0.0838.[cite:94]

These very poor top-1/NLL numbers are consistent with the idea that humans less often choose strategically bad transitions, so such moves may be sparse, heterogeneous, or tied to idiosyncratic mistakes. Interestingly, ECE is lower in these hard degraded-transition slices, which does not mean the model is better overall there; it only means its confidence is numerically closer to observed accuracy, likely because the model is less sharply confident in these difficult cases.[cite:94][cite:83][cite:92]

The `unlabelled` slice sits in between, with NLL 1.8709, Brier 0.7501, top-1 37.53%, top-3 68.33%, top-5 83.11%, and ECE 0.1790.[cite:94] This likely reflects a mixed bag of positions not cleanly assigned to the other transition classes, so intermediate performance is unsurprising.[cite:94]

## Additional assessments and baselines
The `game_val_only` slice scores NLL 1.5709, Brier 0.6750, top-1 44.94%, top-3 76.53%, and top-5 88.28%, which is close to the overall result but with slightly weaker shortlist metrics and slightly lower ECE at 0.1730.[cite:94] This suggests the model generalises similarly on the game-level validation subset, though with marginally noisier ranking quality.[cite:94]

The empirical baseline is not directly comparable to the full model because it is evaluated only on supported states with minimum support 10, covering 4,673 samples and 278,267 events.[cite:94] On that supported subset it achieves NLL 1.3203, top-1 55.85%, top-3 85.05%, top-5 93.34%, and ECE 0.2345.[cite:94] That tells two stories: repeated historical states contain exploitable empirical move frequencies, but those frequencies are themselves not especially well calibrated as probabilities in this metric setup.[cite:94][cite:83]

The reported mean KL on empirical-supported samples is 0.5522.[cite:94] Interpreted qualitatively, that indicates a noticeable but not extreme divergence between the model distribution and the empirical distribution on states with enough support, meaning the learned policy is approximating but not simply copying the observed frequency table.[cite:94]

## Assessment summary
The strongest conclusion is that the model is a good **human move ranker**. On held-out data it usually includes the real human move among its top few options, and its improvements over the uniform baseline are large across all primary metrics.[cite:94]

The main weakness is calibration and exactness in ambiguous settings. Placement-phase positions, high-branching positions, and strategically degrading transitions all show that the model struggles most when human choice is either more stylistic, more diverse, or more weakly determined by local tactical features.[cite:94]

For practical use, the model looks strong enough for candidate pruning, behaviour imitation, or as a prior over human-like play. If the next objective is stronger probability quality rather than just ranking quality, the most promising follow-up work would be calibration-focused evaluation and methods such as temperature scaling, slice-specific calibration analysis, and feature improvements targeted at placement decisions and high-branching states.[cite:94][cite:83][cite:92]
