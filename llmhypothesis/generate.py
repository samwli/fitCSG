from TypeChat.typechat.typechat import TypeChat
import json

def generateTree(tns, object, hypothesis=None):
    # Load the prompt from the prompt.md file
    prompt = None
    with open("prompt.md", "r") as f:
        prompt = f.read()

    ## Replace the $$$ with the actual object name
    prompt = prompt.replace("<ObjectName>", object)

    request = [
        {"role": "user", "content": prompt}
    ]
    if hypothesis:
        request.append({"role": "assistant", "content": hypothesis})
        request.append({"role": "user", "content": f"Can you please generate another tree that uses different shapes and/or operations for the {object} object?"})

    response = tns.translate(request, image=None, return_query=False)   
    if response.success:
        csg = json.loads(str(response.data).replace("'", "\""))
        fn = f"csg_{object}_1.json" if not hypothesis else f"csg_{object}_2.json"
        with open(fn, "w") as f:
            f.write(json.dumps(csg, indent=4))
        return json.dumps(csg, indent=4)
    else:
        print(response.error)

def main():
    # Setup Typechat
    ts = TypeChat()
    ts.createLanguageModel(
        model="gpt-4o", 
        # model="o1-preview", 
        api_key="API_KEY", 
        org_key=None,
        use_json_mode=True,
        user_mode_only=True,
        # base_url = 'http://localhost:23100/v1',
        # model = "llama3.1:70b-instruct-fp16",
    )
    ts.loadSchema("./csgschema.ts")
    tns = ts.createJsonTranslator(name="CSGResponse", basedir="./TypeChat/typechat/schemas")

    # Query the model
    answer_1 = generateTree(tns, object="Mug")
    answer_2 = generateTree(tns, object="Mug", hypothesis=answer_1)

if __name__ == "__main__":
    main()