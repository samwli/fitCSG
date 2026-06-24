# LLM Hypothesis Generation

To generate a json formatted CSG hypothesis, run 

```
python generate.py
```

This will create a JSON file called csg_<object>.json. To change the object, you can put a different target object into lines 47/49 in generate.py. You also need to put your API key into line 36. For now, I would suggest to mostly try with GPT-4o and use o1-preview only sometimes to test how good we can really get as using it is pretty expensive. However, GPT-4o is reasonably cheap, so you can use it for debugging. You can (try to) visualize it with

```
python viz.py csg_<object>.json
```

The visualizer isn't super stable, so sometimes it just gives errors even though the csg is valid. That may be something to look into. 

## Changing prompts

The main (first) prompt is in prompt.md. The follow-up one should probably ne another md file too, but is currently in generate.py line 18. 

# TypeChat
You will need to look at the documentation for the TypeChat module too such that it can start working. I am using it here to force a particular output format for the CSG tree. Among other things, you will need a typescript compile to verify the format. 