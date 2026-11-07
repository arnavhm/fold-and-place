# Demo recording script (about 6 minutes)

Record the screen with **QuickTime → File → New Screen Recording** (or ⌘⇧5), browser at full
screen, zoom 90%. Speak the lines in *italics* or paraphrase them.

## Before recording

```bash
cd fold-and-place
make app            # opens http://localhost:8501
```

* Open the **Plan** page once and press **Solve and compare with baselines** so the solver is warm
  (about 30 s on the first run), then go back to **Home**.
* Close other tabs; turn on Do Not Disturb.

## 0:00 - Home (30 s)

*"Fold & Place answers three questions for a family's shared wardrobe: how to fold each garment,
where to put it, and how to share the space fairly when everyone says their clothes matter most.
It has four stages: see, fold, place, share."*

Point at the four headline numbers.

## 0:30 - Wardrobe (30 s)

Open **Wardrobe**. Switch the preset from *2-door* to *3-door*, then back.

*"These are typical Indian almirah layouts: loft on top, hanging rail, shelves, drawers. The yellow
band is the golden zone - the height you can reach without bending or a stool. Each person's upper
limit is their own shoulder height."*

## 1:00 - Garments (45 s)

Open **Garments**. Scroll the table briefly.

*"The demo family - a student, a mother and a father - owns 97 garments, Indian and Western:
sarees, kurtis, a lehenga, jeans, shirts. Values and wear rates are stated assumptions."*

Switch to **Add from a photo** → *Sample* → `kurti_M.png` → **Analyse**.

*"From a photo, a fashion-tuned vision model classifies the garment, a segmenter cuts it out, and
the A4 sheet in the frame gives real-world scale through a homography. Here the true size is
known, so you can see the error."*
(If the vision models are not installed, say: *"This needs the vision install; the measurements
are evaluated in experiment E3."* and move on.)

## 1:45 - Plan (75 s)

Open **Plan**, press **Solve and compare with baselines**.

*"Four rules of thumb - traditional flat stacks, KonMari filing, hang everything, and a greedy rule -
against an exact constraint-programming model. Every garment gets one fold and one place; the
objective stores as much value as possible, then puts frequently worn clothes in easy reach, then
avoids creasing delicate fabrics."*

Point at the metrics: *"Same 97 garments stored, about a quarter less effort to reach things and
about two-thirds fewer creases than flat stacks - and the solver proves it is optimal."*

Click a saree row in the table: *"Every placement comes with a reason. Let's watch this one fold."*

## 3:00 - Fold studio (60 s)

Open **Fold studio** (the selected garment is preselected; the plan's fold is marked ★).

Let the animation play. *"The fold is exact polygon geometry: each step turns part of the garment
over a crease. The packet size, number of plies and creases feed the optimizer."*

Switch to **Any category** → *Saree*, play it: *"A saree is halved lengthwise, then accordion-folded
into panels."* Drag to orbit once.

## 4:00 - Mechanism lab (90 s)

Open **Mechanism lab**. Expand *The scenario* for a second.

*"Now the game theory. Three people share one small section and not everything fits. Each knows
how much their own clothes matter - a private value - and reports a number. The mechanism decides."*

Press **Run** with everyone truthful. *"VCG picks the welfare-maximising plan and charges each
person the harm they do to the others. Random serial dictatorship uses no money. Maximum Nash
welfare targets fairness."*

Move **Student claims x true value** to **3.0**, press **Run**.

*"Now the student triples every value. Under VCG the lie costs them twelve points; under 'priority
by claimed value' - what a family would naturally do - it gains them twenty-three. That is why the
mechanism matters."*

## 5:30 - Experiments (30 s)

Open **Experiments**, click through E4, E5, E6.

*"Seven experiments back the numbers: fold calibration, measurement error, optimizer against
rules of thumb, mechanism properties under thousands of lies, Shapley cost sharing for buying a
shared wardrobe, and the price of anarchy when everyone games the system. All code and results are
on GitHub."*

## After recording

Trim the start and end in QuickTime (⌘T), export 1080p, and upload it (YouTube unlisted or Drive)
to link from the submission.
