import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompts = [
    "Interview about technology: ",
    "Poem about the sea: ",
    "First sentence of a mystery novel: ",
    "Quote from a scientist on discovery: ",
    "Dialogue in a coffee shop: ",
    "Thoughts on happiness: ",
    "Advice for young artists: ",
    "Reflections on time travel: ",
    "A day in the life of a pilot: ",
    "Debate on the importance of education: ",
    "Book sentence about ancient civilizations: ",
    "Monologue from a Shakespearean character: ",
    "Letter from a soldier during the war: ",
    "Journal entry on a rainy day: ",
    "Recipe description for a festive meal: ",
    "News headline in 2050: ",
    "Travelogue excerpt about Paris: ",
    "Fantasy description of a magical kingdom: ",
    "Review of a futuristic gadget: ",
    "Commentary on social media trends: ",
    "Haiku about autumn: ",
    "Opening lines of a speech on climate change: ",
    "Biography snippet of a famous inventor: ",
    "Dialogues between two old friends meeting after years: ",
    "A philosophical question on the nature of reality: ",
    "A child’s perspective on the world: ",
    "A villain’s justification for their actions: ",
    "An argument in favor of space exploration: ",
    "A description of a peaceful morning in the countryside: ",
    "A suspenseful build-up to a surprising event: ",
    "Interview about technology: ",
    "Poem about the sea: ",
    "First sentence of a mystery novel: ",
    "Quote from a scientist on discovery: ",
    "Dialogue in a coffee shop: ",
    "Thoughts on happiness: ",
    "Advice for young artists: ",
    "Reflections on time travel: ",
    "A day in the life of a pilot: ",
    "Debate on the importance of education: ",
    "Book sentence about ancient civilizations: ",
    "Monologue from a Shakespearean character: ",
    "Letter from a soldier during the war: ",
    "Journal entry on a rainy day: ",
    "Recipe description for a festive meal: ",
    "News headline in 2050: ",
    "Travelogue excerpt about Paris: ",
    "Fantasy description of a magical kingdom: ",
    "Review of a futuristic gadget: ",
    "Commentary on social media trends: ",
    "Haiku about autumn: ",
    "Opening lines of a speech on climate change: ",
    "Biography snippet of a famous inventor: ",
    "Dialogues between two old friends meeting after years: ",
    "A philosophical question on the nature of reality: ",
    "A child’s perspective on the world: ",
    "A villain’s justification for their actions: ",
    "An argument in favor of space exploration: ",
    "A description of a peaceful morning in the countryside: ",
    "A suspenseful build-up to a surprising event: ",
    "Eulogy for a beloved pet: ",
    "Instructions for a time traveler: ",
    "A secret revealed at a family gathering: ",
    "The discovery of a new planet: ",
    "A message in a bottle found on the beach: ",
    "The last entry in a captain's log: ",
    "A wish made upon a shooting star: ",
    "An unexpected friendship between a cat and a mouse: ",
    "A debate between heart and mind: ",
    "A diary entry from a cabin in the woods: ",
    "The inauguration speech of the first Mars colony leader: ",
    "A love letter from the future: ",
    "The first law of a utopian society: ",
    "A beginner's guide to meditation: ",
    "A chef's note on the perfect cup of coffee: ",
    "A detective's clue leading to a breakthrough: ",
    "A ghost's lament: ",
    "A hacker's manifesto: ",
    "An alien's observations about Earth: ",
    "A child's first encounter with snow: ",
    "A farewell letter before a great adventure: ",
    "The founding principles of a secret society: ",
    "A villager's account of a festival: ",
    "A scientist's eureka moment: ",
    "A monk's insights into inner peace: ",
    "A critique of modern art: ",
    "An explorer's encounter with a lost civilization: ",
    "A gardener's philosophy: ",
    "A pirate's code of honor: ",
    "A soldier's promise: ",
    "A witch's recipe for a love potion: ",
    "A journalist's report from a dystopian future: ",
    "A CEO's vision for the next 100 years: ",
    "A composer's inspiration for a symphony: ",
    "A conspiracy theorist's warning: ",
    "A gamer's strategy for the ultimate quest: ",
    "A fan's letter to a fictional character: ",
    "A historian's prediction for the next century: ",
    "An astronaut's first impressions of alien life: ",
    "A philosopher's debate with a computer: ",
    "A traveler's tale of a city in the clouds: ",
    "A critic's review of an interstellar restaurant: ",
    "A teacher's advice for lifelong learning: ",
    "A student's dream of future technology: ",
    "A fashion designer's concept for wearable tech: ",
    "A biologist's discovery of a new species: ",
    "A poet's ode to the midnight sun: ",
    "A programmer's algorithm for happiness: ",
    "A spy's guide to invisibility: ",
    "A firefighter's moment of bravery: ",
    "A runner's thoughts at the starting line: ",
    "A mechanic's appreciation for classic cars: ",
    "A pilot's view above the clouds: ",
    "A sailor's respect for the sea: ",
    "An actor's breakthrough role: ",
    "A director's vision for a film about peace: ",
    "A dancer's interpretation of a storm: ",
    "A sculptor's muse: ",
    "A photographer's quest for the perfect shot: ",
    "A critic's review of an interstellar restaurant: ",
    "A teacher's advice for lifelong learning: ",
    "A student's dream of future technology: ",
    "A fashion designer's concept for wearable tech: ",
    "A biologist's discovery of a new species: ",
    "A poet's ode to the midnight sun: ",
    "A programmer's algorithm for happiness: ",
    "A spy's guide to invisibility: ",
    "A firefighter's moment of bravery: ",
    "A runner's thoughts at the starting line: ",
    "A mechanic's appreciation for classic cars: ",
    "A pilot's view above the clouds: ",
    "A sailor's respect for the sea: ",
    "An actor's breakthrough role: ",
    "A director's vision for a film about peace: ",
    "A dancer's interpretation of a storm: ",
    "A sculptor's muse: ",
    "A photographer's quest for the perfect shot: ",
    "A traveler's discovery of an enchanted forest: ",
    "A writer's struggle with writer's block: ",
    "A climber's conquest of a mountain: ",
    "A scientist's theory on parallel universes: ",
    "A doctor's notes on empathy: ",
    "A farmer's almanac for the future: ",
    "An engineer's blueprint for a sustainable city: ",
    "A politician's promise for a better world: ",
    "A librarian's favorite book: ",
    "A child's imagination of outer space: ",
    "A baker's secret ingredient: ",
    "A florist's language of flowers: ",
    "A mail carrier's unexpected delivery: ",
    "A coach's pep talk before the big game: ",
    "A detective's deduction in a cold case: ",
    "A knight's vow to their kingdom: ",
    "A wizard's first spell gone awry: ",
    "A vampire's reflection on immortality: ",
    "A zombie's diary: ",
    "A mythical creature's hidden life: ",
    "The secret behind an ancient magic spell: ",
    "A day in the life of a robot: ",
    "The true story of a haunted house: ",
    "Underwater civilizations and their mysteries: ",
    "The diary of a space explorer: ",
    "A superhero's dilemma: ",
    "A villain's unlikely redemption: ",
    "The unexpected benefits of time travel: ",
    "Surviving in a post-apocalyptic world: ",
    "The art of potion making: ",
    "The politics of fairy tale kingdoms: ",
    "A detective story set in the future: ",
    "A utopia where nature and technology merge: ",
    "The challenges of life on Mars: ",
    "The life of a pirate in the Caribbean: ",
    "A love story between stars: ",
    "The legend of a lost city: ",
    "Adventures in a virtual reality world: ",
    "A tale of two rival magicians: ",
    "The founding of a new planet: ",
    "Secrets of the deep ocean: ",
    "The last dragon on Earth: ",
    "A journey through the multiverse: ",
    "The rise of artificial intelligence: ",
    "The hidden world of shadows: ",
    "An epic quest for a legendary artifact: ",
    "Life inside a computer game: ",
    "The consequences of a global blackout: ",
    "A society ruled by animals: ",
    "The discovery of a parallel universe: ",
    "A rebellion in a dystopian society: ",
    "A romance that transcends time: ",
    "The mystery of the Bermuda Triangle: ",
    "Survival strategies in a zombie apocalypse: ",
    "The ethics of human cloning: ",
    "A war between humans and aliens: ",
    "The secrets of the Illuminati: ",
    "Exploring the ruins of ancient civilizations: ",
    "The quest for the fountain of youth: ",
    "A world where magic is real: ",
    "The invention that changed the future: ",
    "A journey to the center of the Earth: ",
    "The end of the universe: ",
    "Life as a gladiator in ancient Rome: ",
    "The curse of an ancient pharaoh: ",
    "A spy mission gone wrong: ",
    "The colonization of the solar system: ",
    "A heist in a futuristic city: ",
    "The life of a ghost hunter: ",
    "A scandal in a galactic empire: ",
    "The first contact with an alien species: ",
    "Secrets of a time traveler: ",
    "The adventures of a space pirate: ",
    "The creation of a new galaxy: ",
    "A world dominated by dragons: ",
    "Escaping from a maximum-security prison on Mars: ",
    "A society where dreams can be controlled: ",
    "The downfall of a superhero: ",
    "The revival of extinct species: ",
    "A conspiracy to overthrow a government: ",
    "The discovery of a new form of life: ",
    "A reality show in outer space: ",
    "The challenges of interstellar travel: ",
    "A world powered by magic: ",
    "The aftermath of a nuclear war: ",
    "A journey with a time machine: ",
    "The secrets of an ancient manuscript: ",
    "A battle in a cyberpunk city: ",
    "The ethics of mind control: ",
    "A romance in a steampunk world: ",
    "The legend of a powerful sorcerer: ",
    "A rebellion against a tyrant in a fantasy world: ",
    "Discovering an underground civilization: ",
    "The creation of a utopian society: ",
    "A thriller set in an abandoned space station: ",
    "The mystery of a missing spaceship: ",
    "The challenges of living on an alien planet: ",
    "A fantasy world where seasons last for centuries: ",
    "A crime story in a world with no laws: ",
    "The impact of an asteroid on ancient Earth: ",
    "A journey to find a mythical island: ",
    "The story of a civilization living in the clouds: ",
    "A world where books are banned: ",
    "The rise and fall of a digital empire: ",
    "A quest to decode an alien message: ",
    "The secrets of a mysterious forest: ",
    "A thriller involving an ancient conspiracy: ",
    "A world where humans and machines merge: ",
    "The exploration of a new dimension: ",
    "A society on the brink of a technological singularity: ",
    "A battle for control of a powerful energy source: ",
    "The adventures of a band of space explorers: ",
    "The consequences of altering history: ",
    "A civilization living on a dyson sphere: ",
    "The struggle for freedom in a virtual world: ",
    "A world where emotions are controlled by technology: ",
    "The quest for eternal life: ",
    "Surviving in a world overrun by giant insects: ",
    "The rise of a new species: ",
        "The secret life of an undercover robot: ",
    "A message hidden in ancient runes: ",
    "The diary of a wizard living among humans: ",
    "A breakthrough in teleportation technology: ",
    "A chilling prophecy from an old book: ",
    "The last tree on Earth speaks: ",
    "A virtual reality world that feels more real than life: ",
    "The ethics of creating artificial emotions: ",
    "A rebellion in a world controlled by corporations: ",
    "The discovery of a hidden underwater city: ",
    "An artist who can paint the future: ",
    "A world where music is the source of power: ",
    "The consequences of a global ban on the internet: ",
    "A love story between the sun and the moon: ",
    "A cookbook for intergalactic dishes: ",
    "A heist to steal the most famous painting in the galaxy: ",
    "A planet where shadows live separate lives: ",
    "Surviving in a world without sleep: ",
    "The invention of a device that translates animal thoughts: ",
    "A society obsessed with reversing aging: ",
    "The founding of the first underwater nation: ",
    "A documentary filmmaker in a dystopian future: ",
    "The rise of a new sport that defies gravity: ",
    "A world where everyone can read minds: ",
    "The hidden dangers of parallel universe exploration: ",
    "A reality TV show about colonizing Mars: ",
    "The secret organization that controls the weather: ",
    "A world where history books are banned: ",
    "The last library in the world: ",
    "A mission to save a dying star: ",
    "A city that moves on giant wheels: ",
    "A society where currency is based on creativity: ",
    "A device that allows you to live inside books: ",
    "The mystery of a city that appears only at night: ",
    "A world governed by artificial intelligence: ",
    "The creation of the first true AI artist: ",
    "The lost technology of ancient civilizations: ",
    "A journey to the edge of the universe: ",
    "A world where dreams can be shared: ",
    "The quest for the ultimate source of knowledge: ",
    "A civilization living on the back of a giant creature: ",
    "The discovery of a cure for loneliness: ",
    "A reality where video games become reality: ",
    "The unintended consequences of a global peace treaty: ",
    "A device that allows you to change your appearance at will: ",
    "A rebellion against a future where emotions are regulated: ",
    "The first human to be born on another planet: ",
    "A world where plants are the dominant species: ",
    "The invention of a time-stopping machine: ",
    "A secret society of people who live in the clouds: ",
    "The return of magic to a modern world: ",
    "A space elevator to the stars: ",
    "The discovery of an immortal being: ",
    "A world where water is more valuable than gold: ",
    "The consequences of a world without death: ",
    "A journey through a black hole: ",
    "The creation of a new universe in a lab: ",
    "A society built around the remnants of old technology: ",
    "The last human colony in a post-apocalyptic world: ",
    "A world where memories can be bought and sold: ",
    "The moral dilemmas of genetic editing: ",
    "An expedition to a newly discovered planet: ",
    "A future where humans have evolved into different species: ",
    "The discovery of a parallel world where dinosaurs never went extinct: ",
    "A machine that can bring fictional characters to life: ",
    "A society where lying is physically impossible: ",
    "The impact of discovering we're not alone in the universe: ",
    "A world where everyone is blind: ",
    "The final days of Earth before moving to a new planet: ",
    "A society built on the ruins of an ancient advanced civilization: ",
    "The quest to find the last untouched place on Earth: ",
    "A global event that erases all electronic data: ",
    "The rise of a new human sense: ",
    "A world where you can customize the weather: ",
    "The struggle of living on a planet with extreme seasons: ",
    "The discovery of a new form of communication with the universe: ",
    "A society that has abolished all forms of government: ",
    "The creation of a portal to other dimensions: ",
    "A world where aging can be paused: ",
    "The discovery of an ancient artifact that grants wishes: ",
    "The first contact with a benevolent alien species: ",
    "A world where art is the only currency: ",
    "The rise and fall of a civilization on Jupiter's moon: ",
    "The discovery of a planet identical to Earth: ",
    "A world where humans can photosynthesize: ",
    "The invention of a universal language: ",
    "A society that lives in complete darkness: ",
    "The ethical implications of memory manipulation: ",
    "A journey to reclaim a lost heritage on a distant planet: ",
    "A world where time flows backwards: ",
    "The creation of the first sentient AI city: ",
    "A society where every citizen must pass a test to become an adult: ",
    "The discovery of an energy source that changes everything: ",
    "A world where you can see the sounds: ",
    "The colonization of the ocean floor: ",
    "A rebellion in a technologically advanced utopia: ",
    "The emergence of a new human ability: ",
    "A world where books are the most prized possessions: ",
    "The consequences of discovering the secret to immortality: ",
    "A society built on the back of a giant, living creature: ",
    "The first successful mission to the core of the Earth: ",
    "A world where humans coexist with their digital clones: ",
    "The discovery of a hidden world beneath the ice of Antarctica: ",
    "A journey to rescue a lost expedition in a parallel universe: ",
    "The unveiling of an ancient civilization's greatest invention: ",
    "A world where emotions have physical forms: ",
    "The quest for a mythical land said to cure all diseases: ",
    "A society that has mastered the art of living underwater: ",
    "The creation of a perfect virtual reality: ",
    "The discovery of a hidden door to a forgotten realm: ",
    "A world where every person has a guardian spirit: ",
    "The invention of a machine that can alter reality: ",
    "A society that uses dreams to predict the future: ",
    "The last voyage of a spaceship destined for a black hole: ",
    "A world where magic and technology are indistinguishable: ",
    "The discovery of an alien ship buried on the moon: ",
    "A journey into the heart of a dying star: ",
    "A society where knowledge is the only form of currency: ",
    "The invention of a device that allows travel between alternate realities: ",
    "A world where silence is sacred: ",
    "The quest to build the first city in space: ",
    "The challenges of forming a new society on a distant planet: ",
    "A future where Earth is a museum visited by aliens: ",
    "The ethical dilemmas faced by the first generation of time travelers: ",
    "A world where humans have the ability to teleport: "
]

def generate_sentence(prompt, min_words=20, max_words=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    num_tokens_in_prompt = input_ids.shape[1]

    # Generate a sequence of tokens
    output_sequences = model.generate(
    input_ids=input_ids,
    max_length=50,
    temperature=0.6, 
    top_k=40, 
    top_p=0.9,  
    do_sample=True,       
    num_return_sequences=1
)
    
    generated_text = tokenizer.decode(output_sequences[0][num_tokens_in_prompt:], skip_special_tokens=True)

    words = generated_text.split()
    if len(words) > max_words:
        words = words[:max_words]
    elif len(words) < min_words:
        print("too short")
        return None  # Skip if the sentence is too short

    return ' '.join(words)
import random
output_file = "AI_sentences.txt"
with open(output_file, 'w') as file:
    for prompt in prompts:
        sentence = generate_sentence(prompt+" in about 40 words")
        if sentence:
            file.write(sentence + '\n')

