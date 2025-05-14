import json
import os
import re
import random

import inflect
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pkgs.openai.clip import load as load_model

def generate_word_forms(keyword):
    p = inflect.engine()
    forms = set()

    # Processing singular complex conversion
    singular = p.singular_noun(keyword)
    if singular:
        forms.update([singular, keyword])
    else:
        forms.add(keyword)
        plural = p.plural(keyword)
        if plural != keyword:  # Avoid duplicate additions
            forms.add(plural)

    escaped_forms = [re.escape(form) for form in forms]
    return rf'(?i)\b({'|'.join(escaped_forms)})\b' 

def filter_with_pandas(keyword,df):

    # Generate regular expressions
    patterns = generate_word_forms(keyword)

    filtered_df = df[df["caption"].str.contains(
        patterns, case=False, regex=True, na=False
    )]

def remove_word(p,sentence, word_to_remove):
    # Obtain the plural form
    plural_word = p.plural(word_to_remove)

    # Match singular or plural forms and ensure that unnecessary spaces are removed when deleting
    pattern = r'\b' + re.escape(word_to_remove) + r's?\b|\b' + re.escape(plural_word) + r'\b'
    sentence = re.sub(pattern, '', sentence).strip()
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

def get_distance(feature1, feature2, logit_scale=100):
    # normalized features
    image_features = feature1 / feature1.norm(dim=1, keepdim=True)
    text_features = feature2 / feature2.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = logit_scale if isinstance(logit_scale, int) else logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    return logits_per_image, logits_per_text

def save_category_embeddings(model,processor,category_list,save_path,templates):
    model.eval()
    device = next(model.parameters()).device
    embeddings_dict = {}

    for category in tqdm(category_list):
        with torch.no_grad():
            texts = [template(category) for template in templates]
            caption = processor.process_text(texts)
            input_ids, attention_mask = caption['input_ids'].to(device), caption['attention_mask'].to(device)
            text_embedding = model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()

            embeddings_dict[category] = text_embedding.cpu().detach()

    torch.save(embeddings_dict, save_path)
    print(f"Successfully saved category embedding into: {save_path}")


def get_text_embedding(texts,processor,model,device):
    # global caption, text_embedding
    caption = processor.process_text(texts)
    input_ids, attention_mask = caption['input_ids'].to(device), caption['attention_mask'].to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]

if __name__ == '__main__':
    # load model
    device = "cuda:0"
    model, processor = load_model(name='ViT-B/32', pretrained=True)
    model.to(device)
    model.eval()

    target_class = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark', 'electric ray', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', 'bulbul', 'jay', 'magpie', 'chickadee', 'american dipper', 'bald eagle', 'vulture', 'great grey owl', 'fire salamander', 'newt', 'spotted salamander', 'axolotl', 'american bullfrog', 'tree frog', 'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana', 'agama', 'alligator lizard', 'gila monster', 'european green lizard', 'chameleon', 'komodo dragon', 'nile crocodile', 'american alligator', 'triceratops', 'eastern hog-nosed snake', 'kingsnake', 'garter snake', 'water snake', 'vine snake', 'boa constrictor', 'african rock python', 'indian cobra', 'green mamba', 'sea snake', 'eastern diamondback rattlesnake', 'sidewinder rattlesnake', 'trilobite', 'scorpion', 'yellow garden spider', 'european garden spider', 'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'peafowl', 'quail', 'partridge', 'african grey parrot', 'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'duck', 'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'dungeness crab', 'rock crab', 'fiddler crab', 'red king crab', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'great egret', 'crane bird', 'american coot', 'bustard', 'ruddy turnstone', 'dunlin', 'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong', 'sea lion', 'chihuahua', 'japanese chin', 'maltese', 'pekingese', 'shih tzu', 'papillon', 'toy terrier', 'rhodesian ridgeback', 'afghan hound', 'basset hound', 'beagle', 'bloodhound', 'bluetick coonhound', 'black and tan coonhound', 'treeing walker coonhound', 'english foxhound', 'redbone coonhound', 'borzoi', 'irish wolfhound', 'italian greyhound', 'whippet', 'ibizan hound', 'norwegian elkhound', 'saluki', 'scottish deerhound', 'weimaraner', 'staffordshire bull terrier', 'american staffordshire terrier', 'bedlington terrier', 'border terrier', 'kerry blue terrier', 'irish terrier', 'norfolk terrier', 'norwich terrier', 'yorkshire terrier', 'wire fox terrier', 'lakeland terrier', 'sealyham terrier', 'airedale terrier', 'cairn terrier', 'dandie dinmont terrier', 'boston terrier', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'scottish terrier', 'tibetan terrier', 'australian silky terrier', 'soft-coated wheaten terrier', 'west highland white terrier', 'lhasa apso', 'flat-coated retriever', 'golden retriever', 'labrador retriever', 'chesapeake bay retriever', 'german shorthaired pointer', 'vizsla', 'english setter', 'irish setter', 'gordon setter', 'clumber spaniel', 'english springer spaniel', 'cocker spaniel', 'sussex spaniel', 'irish water spaniel', 'kuvasz', 'schipperke', 'malinois', 'australian kelpie', 'komondor', 'old english sheepdog', 'shetland sheepdog', 'collie', 'border collie', 'rottweiler', 'german shepherd dog', 'dobermann', 'miniature pinscher', 'greater swiss mountain dog', 'bernese mountain dog', 'boxer', 'bullmastiff', 'tibetan mastiff', 'french bulldog', 'great dane', 'st. bernard', 'husky', 'alaskan malamute', 'siberian husky', 'dalmatian', 'basenji', 'pug', 'leonberger', 'newfoundland dog', 'great pyrenees dog', 'samoyed', 'pomeranian', 'chow chow', 'keeshond', 'brussels griffon', 'pembroke welsh corgi', 'cardigan welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'grey wolf', 'coyote', 'dingo', 'dhole', 'african wild dog', 'hyena', 'red fox', 'kit fox', 'arctic fox', 'grey fox', 'tabby cat', 'tiger cat', 'persian cat', 'siamese cat', 'egyptian mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'american black bear', 'polar bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket insect', 'stick insect', 'cockroach', 'praying mantis', 'cicada', 'lacewing', 'dragonfly', 'damselfly', 'red admiral butterfly', 'monarch butterfly', 'sulphur butterfly', 'starfish', 'sea urchin', 'sea cucumber', 'cottontail rabbit', 'hare', 'angora rabbit', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'bighorn sheep', 'alpine ibex', 'hartebeest', 'gazelle', 'llama', 'weasel', 'mink', 'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'patas monkey', 'baboon', 'macaque', 'langur', 'proboscis monkey', 'marmoset', 'white-headed capuchin', 'howler monkey', 'titi monkey', 'common squirrel monkey', 'ring-tailed lemur', 'asian elephant', 'african bush elephant', 'red panda', 'giant panda', 'eel', 'silver salmon', 'clownfish', 'sturgeon', 'gar fish', 'lionfish', 'pufferfish', 'abacus', 'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance', 'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'trash can', 'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint pen', 'band-aid', 'banjo', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap', 'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker', 'beer bottle', 'beer glass', 'bell tower', 'baby bib', 'tandem bicycle', 'bikini', 'ring binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie', 'bookcase', 'bookstore', 'bottle cap', 'hunting bow', 'bow tie', 'bra', 'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train', 'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe', 'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit', 'car wheel', 'automated teller machine', 'cassette', 'cassette player', 'castle', 'catamaran', 'cd player', 'cello', 'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw', 'storage chest', 'china cabinet', 'christmas stocking', 'church', 'movie theater', 'cleaver', 'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug', 'coffeemaker', 'combination lock', 'computer keyboard', 'candy store', 'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'construction crane', 'crash helmet', 'crate', 'infant bed', 'crock pot', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'rotary dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table', 'dishwasher', 'disc brake', 'dock', 'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick', 'dumbbell', 'dutch oven', 'electric fan', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso machine', 'face powder', 'feather boa', 'filing cabinet', 'fireboat', 'fire truck', 'fire screen', 'flagpole', 'flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster bed', 'freight car', 'french horn', 'frying pan', 'fur coat', 'garbage truck', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'radiator grille', 'grocery store', 'guillotine', 'hair clip', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer', 'handkerchief', 'hard disk drive', 'harmonica', 'harp', 'combine harvester', 'hatchet', 'holster', 'home theater', 'honeycomb', 'hook', 'hoop skirt', 'gymnastic horizontal bar', 'horse-drawn vehicle', 'hourglass', 'ipod', 'clothes iron', 'carved pumpkin', 'jeans', 'jeep', 't-shirt', 'jigsaw puzzle', 'rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade', 'laptop computer', 'lawn mower', 'lens cap', 'letter opener', 'library', 'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick', 'slip-on shoe', 'lotion', 'music speaker', 'sawmill', 'magnetic compass', 'messenger bag', 'mailbox', 'tights', 'manhole cover', 'maraca', 'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine cabinet', 'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home', 'ford model t', 'modem', 'monastery', 'monitor', 'moped', 'mortar and pestle', 'graduation cap', 'mosque', 'mosquito net', 'vespa', 'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van', 'muzzle', 'metal nail', 'neck brace', 'necklace', 'baby pacifier', 'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'pipe organ', 'oscilloscope', 'bullock cart', 'oxygen mask', 'paddle', 'paddle wheel', 'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'railroad car', 'patio', 'payphone', 'pedestal', 'pencil case', 'pencil sharpener', 'perfume', 'petri dish', 'photocopier', 'plectrum', 'pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship', 'block plane', 'planetarium', 'plastic bag', 'plate rack', 'plunger', 'polaroid camera', 'pole', 'police van', 'poncho', 'pool table', 'soda bottle', 'plant pot', "potter's wheel", 'power drill', 'prayer rug', 'printer', 'prison', 'projector', 'hockey puck', 'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reflex camera', 'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair', 'rotisserie', 'eraser', 'rugby ball', 'sneaker', 'safe', 'safety pin', 'salt shaker', 'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale', 'school bus', 'schooner', 'scoreboard', 'crt monitor', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'sleeping bag', 'slide rule', 'sliding door', 'slot machine', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'sombrero', 'soup bowl', 'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive', 'steel drum', 'stethoscope', 'scarf', 'stone wall', 'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglasses', 'sunscreen', 'suspension bridge', 'mop', 'sweatshirt', 'swing', 'electrical switch', 'syringe', 'table lamp', 'tank', 'tape player', 'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof', 'thimble', 'threshing machine', 'throne', 'tile roof', 'toaster', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'toy store', 'tractor', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus', 'trombone', 'hot tub', 'turnstile', 'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase', 'velvet fabric', 'vending machine', 'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine', 'water bottle', 'water jug', 'water tower', 'whistle', 'hair wig', 'window screen', 'window shade', 'wine bottle', 'airplane wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence', 'shipwreck', 'sailboat', 'yurt', 'website', 'comic book', 'crossword', 'traffic light', 'dust jacket', 'menu', 'plate', 'guacamole', 'hot pot', 'trifle', 'ice cream', 'popsicle', 'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog', 'mashed potatoes', 'cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'granny smith apple', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'pomegranate', 'hay', 'carbonara', 'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito', 'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble', 'cliff', 'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar', 'beach', 'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'coral fungus', 'agaric', 'hen of the woods mushroom', 'bolete', 'corn cob', 'toilet paper']
    templates = [lambda s: f"a bad photo of a {s}.", lambda s: f"a photo of many {s}.",
                 lambda s: f"a sculpture of a {s}.", lambda s: f"a photo of the hard to see {s}.",
                 lambda s: f"a low resolution photo of the {s}.", lambda s: f"a rendering of a {s}.",
                 lambda s: f"graffiti of a {s}.", lambda s: f"a bad photo of the {s}.",
                 lambda s: f"a cropped photo of the {s}.", lambda s: f"a tattoo of a {s}.",
                 lambda s: f"the embroidered {s}.", lambda s: f"a photo of a hard to see {s}.",
                 lambda s: f"a bright photo of a {s}.", lambda s: f"a photo of a clean {s}.",
                 lambda s: f"a photo of a dirty {s}.", lambda s: f"a dark photo of the {s}.",
                 lambda s: f"a drawing of a {s}.", lambda s: f"a photo of my {s}.", lambda s: f"the plastic {s}.",
                 lambda s: f"a photo of the cool {s}.", lambda s: f"a close-up photo of a {s}.",
                 lambda s: f"a black and white photo of the {s}.", lambda s: f"a painting of the {s}.",
                 lambda s: f"a painting of a {s}.", lambda s: f"a pixelated photo of the {s}.",
                 lambda s: f"a sculpture of the {s}.", lambda s: f"a bright photo of the {s}.",
                 lambda s: f"a cropped photo of a {s}.", lambda s: f"a plastic {s}.",
                 lambda s: f"a photo of the dirty {s}.", lambda s: f"a jpeg corrupted photo of a {s}.",
                 lambda s: f"a blurry photo of the {s}.", lambda s: f"a photo of the {s}.",
                 lambda s: f"a good photo of the {s}.", lambda s: f"a rendering of the {s}.",
                 lambda s: f"a {s} in a video game.", lambda s: f"a photo of one {s}.", lambda s: f"a doodle of a {s}.",
                 lambda s: f"a close-up photo of the {s}.", lambda s: f"a photo of a {s}.",
                 lambda s: f"the origami {s}.", lambda s: f"the {s} in a video game.", lambda s: f"a sketch of a {s}.",
                 lambda s: f"a doodle of the {s}.", lambda s: f"a origami {s}.",
                 lambda s: f"a low resolution photo of a {s}.", lambda s: f"the toy {s}.",
                 lambda s: f"a rendition of the {s}.", lambda s: f"a photo of the clean {s}.",
                 lambda s: f"a photo of a large {s}.", lambda s: f"a rendition of a {s}.",
                 lambda s: f"a photo of a nice {s}.", lambda s: f"a photo of a weird {s}.",
                 lambda s: f"a blurry photo of a {s}.", lambda s: f"a cartoon {s}.", lambda s: f"art of a {s}.",
                 lambda s: f"a sketch of the {s}.", lambda s: f"a embroidered {s}.",
                 lambda s: f"a pixelated photo of a {s}.", lambda s: f"itap of the {s}.",
                 lambda s: f"a jpeg corrupted photo of the {s}.", lambda s: f"a good photo of a {s}.",
                 lambda s: f"a plushie {s}.", lambda s: f"a photo of the nice {s}.",
                 lambda s: f"a photo of the small {s}.", lambda s: f"a photo of the weird {s}.",
                 lambda s: f"the cartoon {s}.", lambda s: f"art of the {s}.", lambda s: f"a drawing of the {s}.",
                 lambda s: f"a photo of the large {s}.", lambda s: f"a black and white photo of a {s}.",
                 lambda s: f"the plushie {s}.", lambda s: f"a dark photo of a {s}.", lambda s: f"itap of a {s}.",
                 lambda s: f"graffiti of the {s}.", lambda s: f"a toy {s}.", lambda s: f"itap of my {s}.",
                 lambda s: f"a photo of a cool {s}.", lambda s: f"a photo of a small {s}.",
                 lambda s: f"a tattoo of the {s}."]
    save_path = "category_embeddings_croups.pt"
    # load csv
    csv_path = 'Corpus CSV path' 
    df = pd.read_csv(csv_path)

    p = inflect.engine()
    category_embeddings = torch.load(save_path, map_location=device)

    load_data_batch_size = 1024

    ans_dict={}
    for c in target_class:
        filtered_df = filter_with_pandas(c, df)
        texts = filtered_df['caption'].drop_duplicates().tolist()

        dataloader_texts = batch(texts, load_data_batch_size)
        original_text_embeddings = []
        for text in dataloader_texts:
            original_text_embedding = get_text_embedding(texts=text,processor=processor,model=model,device=device)
            original_text_embeddings.append(original_text_embedding)
        original_text_embeddings = torch.cat(original_text_embeddings, dim=0).to(device)


        bck_infos = []
        for index, row in tqdm(filtered_df.iterrows(), desc=f"Processing {csv_path}"):
            caption = row['caption']
            bck_infos.append(remove_word(p,caption, c))

        dataloader_bck_infos = batch(bck_infos, load_data_batch_size)
        bckinfo_text_embeddings = []
        for text in dataloader_bck_infos:
            bckinfo_text_embedding = get_text_embedding(texts=text, processor=processor, model=model, device=device)
            bckinfo_text_embeddings.append(bckinfo_text_embedding)
        bckinfo_text_embeddings = torch.cat(bckinfo_text_embeddings, dim=0).to(device)

        category_embedding = category_embeddings[c]

        ori_logits = model.logit_scale.exp() * category_embedding @ original_text_embeddings.t()
        ori_logits = ori_logits.mean(dim=0)

        bckinfo_logits = model.logit_scale.exp() * category_embedding @ bckinfo_text_embeddings.t()
        bckinfo_logits = bckinfo_logits.mean(dim=0)

        print(f"{c}\t: ori:{ori_logits.item():.2f}\t bck:{bckinfo_logits.item():.2f}, Difference:{ori_logits.item()-bckinfo_logits.item():.2f}")
        ans_dict[c] = {"ori":f"{ori_logits.item():.2f}","bck":f"{bckinfo_logits.item():.2f}","diff":f"{ori_logits.item()-bckinfo_logits.item():.2f}"}
    # save to json
    output_path = 'out.json' 
    with open(output_path, 'w') as json_file:
        json.dump(ans_dict, json_file, indent=4)