from nsd_access import NSDAccess
import json


def main():
    nsda = NSDAccess('../nsd/')

    prompts = {}
    id = 0

    for p2 in nsda.read_image_coco_info(range(0, 73000),info_type='captions'):
        prompt = []
        
        for p in p2:
            print(p, flush=True)
            prompt.append(p['caption'])    
        
        prompts[id] = prompt
        id += 1

    with open('prompts2.json', 'w') as json_file:
        json.dump(prompts, json_file)

if __name__ == "__main__":
    main()


