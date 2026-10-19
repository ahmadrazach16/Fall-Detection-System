# Prompts for generative video models

Use these to create photo-realistic clips (for example in Veo, Sora, Runway or Kling) that complement the skeleton simulator. Generated footage helps with appearance robustness: clothing, lighting, textures, clutter. It is not physically exact, so treat it as augmentation and never as ground truth.

## Ground rules

* Use fictional, unidentifiable actors only. Never use a real person's likeness.
* Label every clip as synthetic and keep a manifest: prompt, model, version, date, seed.
* Check each clip for plausibility. Generative models often get fall dynamics wrong (floating, slow motion, limb glitches). Reject bad takes.
* Pair every fall clip with non-fall clips under the same camera and lighting, so the detector does not learn the scene instead of the motion.

## Prompt template

```
[Camera]   Fixed CCTV-style camera mounted high in a corner, {angle}, 25 fps, no camera motion.
[Subject]  An elderly person with {hair and clothing}, using a four-wheeled or two-wheeled walker.
[Setting]  {living room | hallway | care-home corridor}, {lighting}.
[Action]   {action description}
[Style]    Realistic documentary footage, natural motion, no cuts, no text, no music.
```

## Fall scenarios

1. **Forward fall onto the walker.** The person shuffles forward behind a walker, the front wheel catches on the carpet edge, they pitch forward, let go of the handles and land on hands and knees, then lie still. Side view.
2. **Backward fall.** The person turns to look behind, loses balance, steps back, sits down heavily and falls onto their back. The walker stays upright. Front-corner view.
3. **Sideways fall.** The person leans to pick up an item from a table, the walker slides sideways and they drop onto their side. Wide shot from above.
4. **Slow slump.** The person stops, appears faint, legs buckle, and they slide down a wall to the floor over about two seconds.

## Hard negatives

5. **Sitting on a chair** after walking to it with the walker.
6. **Bending to pick up an object**, then standing with support from the walker.
7. **Stumble and recover.** A foot catches, the person lurches, grips the walker and regains balance.
8. **Controlled lowering.** The person deliberately kneels, then lies down on a mat for exercise.
9. **Normal walking** at three speeds, **standing still**, **turning with the walker**.

## Variation grid (change one factor at a time)

| Factor | Values |
|---|---|
| Camera yaw | side, 45°, frontal, rear |
| Camera height | 1.5 m, 2.5 m, 3.5 m |
| Lighting | daylight, warm evening lamp, dim night, backlit window |
| Clothing | dark, light, patterned, long skirt, dressing gown |
| Clutter | clear floor, rug, coffee table, pet |
| Occlusion | sofa or table hides the legs for part of the clip |

## Quality-control checklist

- [ ] Duration 8–12 s, with the event starting after 3 s so the detector can calibrate.
- [ ] The person is fully visible for the first second.
- [ ] The descent looks natural (about 0.5–1 s for the fall itself).
- [ ] No identifiable real person, logo or text.
- [ ] Pose estimator confidence is reasonable on the clip.
- [ ] Manifest entry written.
