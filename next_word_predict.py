# import essential libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM




# loading the dataset to the model 
# kaggle dataset used 
# dataset location : https://www.kaggle.com/datasets/ashishpandey2062/next-word-predictor-text-generator-dataset/data

text = [ 
        """The sun was shining brightly in the clear blue sky, and a gentle breeze rustled the leaves of the tall trees. People were out enjoying the beautiful weather, some sitting in the park, others taking a leisurely stroll along the riverbank. Children were playing games, and laughter filled the air.

        As the day turned into evening, the temperature started to drop, and the sky transformed into a canvas of vibrant colors. Families gathered for picnics, and the smell of barbecues wafted through the air. It was a perfect day for a picnic by the lake.

        In the distance, you could hear the sound of live music coming from a local band, and people began to gather around the stage to enjoy the performance. The atmosphere was electric, and the music had everyone swaying to the beat.

        As the stars began to twinkle in the night sky, the crowd grew even larger, and the festivities continued well into the night. It was a day filled with joy, laughter, and memories that would last a lifetime.


        The ancient castle stood on a hill, its towering spires reaching up towards the sky. The castle had a rich history, and its stone walls had witnessed countless battles and royal intrigues. Tourists from all over the world flocked to explore its mysteries.

        Inside the castle, you could find grand halls adorned with magnificent tapestries and chandeliers. The air was thick with the scent of history, and the creaking of old wooden floors echoed in the corridors. The castle's library housed an impressive collection of books, some dating back centuries.

        As you ventured further into the castle, you would discover hidden chambers and secret passages. Legends spoke of a hidden treasure buried somewhere within its walls, waiting to be found by a brave adventurer.

        Outside the castle, a vast moat surrounded it, and a drawbridge provided access to the outside world. Beyond the moat, a lush forest stretched as far as the eye could see, inviting exploration and adventure.

        The village at the base of the hill relied on the castle for protection and trade. The townspeople were friendly and welcoming, and their stories were filled with folklore and local legends.

        At night, the castle's windows lit up with a warm, inviting glow, making it look like something out of a fairy tale. It was a place where history and fantasy intertwined, a place where dreams and reality converged.

        Your RNN model can use this text to predict the next word in a sequence, offering an exciting opportunity for creative text generation and exploration.

        In the heart of the bustling city, the streets were alive with the sounds of traffic and the chatter of people going about their daily lives. Skyscrapers reached towards the heavens, their glass facades reflecting the vibrant energy of the metropolis.

        Street vendors sold a variety of goods, from sizzling hot dogs to handmade jewelry. The aroma of freshly brewed coffee wafted from the corner cafes, where patrons sipped their drinks while watching the world go by.

        Amid the urban chaos, a beautiful park provided a serene escape. Tall trees offered shade, and a tranquil pond was home to ducks and swans. The park's paths were lined with benches where people could sit and read, or simply enjoy the calm in the midst of the urban storm.

        The city's cultural scene was rich and diverse, with theaters showcasing the latest plays and art galleries displaying works from local and international artists. The symphony orchestra filled the air with music, and museums held treasures from various eras.

        At night, the city transformed into a sparkling wonderland. Neon signs illuminated the streets, and restaurants buzzed with diners enjoying cuisine from around the world. Nightclubs and bars beckoned those seeking entertainment and a lively atmosphere.

        In a quaint, picturesque village nestled in a valley between rolling hills, life unfolded at a leisurely pace. The village was a place where time seemed to stand still, where cobblestone streets wound their way through rows of charming cottages adorned with colorful flower boxes.

        The village square was the heart of the community, featuring a centuries-old oak tree where locals gathered to share stories and laughter. A lively market filled with stalls offering fresh produce, handmade crafts, and artisanal treats was a weekly highlight. Children played tag in the square, and the scent of freshly baked bread filled the air.

        The local bakery was renowned for its mouthwatering pastries, and the aroma of buttery croissants and cinnamon rolls lured customers from miles away. The baker, an elderly woman with a twinkle in her eye, had passed down her family recipes for generations.

        Beyond the village, a meandering river sparkled in the sunlight, perfect for lazy summer afternoons of picnicking and swimming. The surrounding meadows were adorned with wildflowers, and the song of birds filled the air.

        Hiking trails led into the lush forests that surrounded the village, providing opportunities for exploration and solitude. Hidden waterfalls, secret glades, and the occasional glimpse of wildlife awaited those who ventured deeper into the woods.

        Nights in the village were enchanting, with the stars in the clear sky casting their glow over the cobblestone streets. Local musicians gathered at the village inn to play folk tunes and serenade the guests, while cozy fireplaces warmed the hearts of those who sought refuge from the cool night air.

        In the vast expanse of the Sahara Desert, the sun blazed relentlessly in the sky, casting a sea of golden dunes as far as the eye could see. The harsh, unforgiving environment was home to a unique and resilient ecosystem, with desert animals and plant life adapted to thrive in this extreme landscape.

        The Sahara, known for its sweeping sand dunes, concealed hidden oases that were a lifeline for travelers and nomadic tribes. These lush pockets of greenery offered shade and sustenance to those braving the desert's scorching heat.

        Nomadic Tuareg tribes roamed the desert on camelback, their indigo-dyed turbans and traditional robes providing protection against the elements. These skilled desert navigators knew the secrets of the dunes, their knowledge passed down through generations.

        At night, the Sahara transformed into a celestial wonderland. The sky was an uninterrupted canvas of stars, and the silence of the desert was only occasionally broken by the call of nocturnal creatures. Campfires flickered, and travelers shared stories under the infinite night sky.

        Further south, the Sahel region transitioned from the arid Sahara into a savanna teeming with wildlife. Elephants, giraffes, and zebras roamed the plains, while lions and cheetahs stealthily hunted their prey. It was a place of beauty and raw, natural power.

        The Sahel was also home to diverse cultures, with vibrant markets and lively celebrations. Colorful textiles and handcrafted jewelry told the stories of the people who created them. Traditional drumming and dancing were integral parts of their cultural heritage.

        Deep in the heart of the Amazon rainforest, a world of enchanting biodiversity unfolded. Towering trees with vast canopies formed a green cathedral that obscured the sun, creating a perpetual twilight on the forest floor. The Amazon was a realm of endless wonders, with millions of species, many of them yet to be discovered.

        Majestic rivers snaked through the dense vegetation, home to piranhas, electric eels, and pink river dolphins. The lush undergrowth concealed colorful poison dart frogs, while howler monkeys and toucans provided the soundtrack of the jungle.

        Indigenous tribes, living in harmony with the rainforest for generations, preserved ancient traditions and knowledge. They used natural remedies from the forest's flora to treat ailments and had a deep spiritual connection with their environment.

        Explorers ventured deep into the Amazon, facing both its beauty and its dangers. Mysterious creatures like the jaguar and the anaconda lurked in the shadows, and the Amazon's diverse birdlife filled the air with their calls and songs.

        At night, the Amazon buzzed with life. Bioluminescent insects and fireflies illuminated the dark, creating a surreal, glowing world. The night chorus of frogs and insects created an orchestra of sounds, while nocturnal animals like the elusive jaguarundi roamed in search of prey.

        The Amazon was a place of awe and wonder, teeming with life and secrets waiting to be uncovered. Your RNN model can use this detailed text to predict the next word in sentences, allowing it to generate rich descriptions of the Amazon rainforest and its incredible biodiversity.

        High in the Himalayan mountains, where the air was thin and frigid, an ancient monastery clung to the edge of a precipitous cliff. The monks who resided there dedicated their lives to meditation, seeking spiritual enlightenment. The monastery was a place of breathtaking natural beauty, with snow-capped peaks, prayer flags fluttering in the wind, and the distant sound of ringing bells.

        Inside, the walls of the monastery were adorned with intricate murals, depicting the life of the Buddha and the legends of their order. A sense of tranquility and mysticism pervaded the air as the monks chanted and prayed.

        The Himalayas, a mountain range often referred to as the "Roof of the World," were home to rare and elusive wildlife. Snow leopards, red pandas, and the majestic golden eagles roamed the high-altitude regions. Rhododendron forests added splashes of vibrant color to the otherwise rugged landscape.

        The local villages, nestled in the valleys, were a testament to human resilience. Their terraced fields produced vital crops, and they traded in yak wool, salt, and other goods. Festivals were celebrated with colorful costumes, lively dances, and traditional music that resonated through the valleys.

        Hiking trails wound through the Himalayas, attracting adventurers from all corners of the globe. Trekkers followed ancient paths that connected remote villages and provided access to some of the world's most breathtaking views. Prayer wheels spun with each passerby's touch, releasing blessings into the world.

        As the sun set behind the Himalayan peaks, the skies were painted with hues of orange and pink. The clear, unpolluted air made stargazing a mesmerizing experience. The Milky Way stretched across the heavens, and constellations told stories in the night sky.

        In the heart of the dense, Amazonian rainforest, an awe-inspiring realm of biodiversity unfolded. Towering trees formed a dense canopy, creating an emerald world beneath where life thrived in countless forms. The Amazon was a place of profound natural wonder, home to millions of species, many of which were yet to be discovered.

        Majestic rivers, like the Amazon and its tributaries, meandered through the dense vegetation, teeming with exotic creatures. Pink river dolphins, piranhas, and electric eels thrived in the murky waters, while the lush undergrowth hid vibrant poison dart frogs and elusive jaguars prowled through the shadows.

        Indigenous tribes, whose roots stretched back through generations, inhabited the rainforest. They possessed a deep understanding of the forest's secrets, relying on its resources for their survival. Their knowledge included the use of medicinal plants and a profound spiritual connection with the natural world.

        Explorers ventured deep into the Amazon, encountering its exquisite beauty and treacherous perils. Mysterious creatures, like the anaconda and harpy eagle, lurked in the wilderness. The cacophony of birdcalls and the resonance of the jungle created an enchanting soundscape.

        At night, the Amazon's bioluminescent insects and fireflies turned the dark into a surreal dreamscape, where the forest seemed to glow. The chorus of frogs and insects created a mesmerizing symphony, while nocturnal animals, including the elusive ocelot, emerged to hunt.

        The Amazon was a realm of magic and mystery, a treasure trove of life and enigma waiting to be unraveled. Your RNN model can use this detailed text to predict the next word in sentences, allowing it to generate vivid and evocative descriptions of the Amazon rainforest and its extraordinary biodiversity.

        On the rugged, wind-swept shores of the Faroe Islands, life unfolded with the resilience and determination of the islanders who called this remote archipelago home. Cliffs rose dramatically from the North Atlantic Ocean, shrouded in mist and battered by the relentless waves. It was a place where nature ruled, where the sea provided sustenance, and the land offered challenges and rewards.

        In the tiny fishing villages, colorful houses with turf roofs nestled among the hills, providing a stark contrast to the rugged landscape. The locals, known for their tenacity, were masterful seafarers who depended on the ocean's bounty. They weathered storms and treacherous waters to catch fish, puffins, and whales.

        The Faroe Islands were also a paradise for birdwatchers and naturalists. The sheer cliffs teemed with seabirds, including puffins, guillemots, and kittiwakes, nesting on narrow ledges. The air was filled with the chorus of birdcalls, creating a symphony unique to this remote corner of the world.

        Explorers ventured to the Faroe Islands, drawn by its untamed beauty. Treacherous hiking trails led to breathtaking vistas, revealing landscapes that seemed otherworldly. Hidden waterfalls cascaded from sheer cliffs, and peaceful lakes nestled among emerald valleys.

        At night, the islands transformed into a realm of tranquility. The low-hanging clouds parted to reveal a star-studded sky, where the Northern Lights danced in vibrant colors. The villages came alive with the glow of cozy, candlelit pubs where locals shared stories and music.

        The Faroe Islands were a place of rugged beauty, where human resilience and the forces of nature intertwined. Your RNN model can use this detailed text to predict the next word in sentences, allowing it to generate vivid descriptions of the Faroe Islands and the unique way of life on these remote isles.

        In the mystical, mist-shrouded valley of Machu Picchu, where ancient ruins nestled amid the Andes, life unfolded with the echoes of the Inca civilization that once thrived there. The stone citadel, perched on a mountaintop, was a testament to human ingenuity and the enchanting beauty of the natural world.

        Machu Picchu was a place of enigmatic history and breathtaking architecture. Stone terraces cascaded down the mountainside, once used for agriculture, while temples and ceremonial plazas stood as a testament to the spiritual practices of the Inca. The surrounding peaks were sacred to the people who once lived there.

        Visitors from around the world embarked on treks to reach Machu Picchu, following the ancient Inca Trail that wound through lush cloud forests and across rickety bridges. The journey was an exploration of the senses, as orchids and colorful butterflies adorned the path, and the sound of rushing rivers provided a soothing background melody.

        Local Quechua people, descendants of the Inca, still inhabited the Andes. They preserved their ancestral traditions and language, and their vibrant textiles and handicrafts told stories of their culture. Festivals celebrated the changing of seasons and the harvest, with colorful parades and music.

        At night, Machu Picchu was a place of serenity. The Milky Way stretched across the heavens, and the ruins were illuminated by the soft glow of lanterns. Visitors meditated on the terraces, gazing out at the starlit landscape and contemplating the mysteries of this ancient city.

        Machu Picchu was a sanctuary of history, a place where the past and present intertwined. Your RNN model can use this detailed text to predict the next word in sentences, allowing it to generate evocative descriptions of Machu Picchu and the mystical allure of this ancient site.

        Deep in the heart of the Icelandic wilderness, where glaciers and volcanoes coexisted, life unfolded amidst the raw power of nature. Rugged lava fields stretched as far as the eye could see, punctuated by geothermal hot springs and bubbling mud pots. It was a land of fire and ice, a place where the elements danced in a mesmerizing symphony.

        Iceland was a place of epic landscapes and geological wonders. Glaciers carved through mountains, and icebergs drifted in icy lagoons. Geysers shot boiling water and steam into the air, erupting with unceasing regularity. Waterfalls cascaded from immense heights, creating rainbows that seemed to touch the ground.

        Local Icelanders, known for their Viking heritage, embraced the unforgiving environment with a spirit of resilience. They relied on their land's abundant geothermal energy to heat their homes and power their greenhouses, where they grew vegetables despite the harsh climate. Festivals celebrated the country's Viking history, with traditional sagas and music.

        Adventurers from around the world journeyed to Iceland, seeking experiences that ranged from ice cave exploration to snorkeling between tectonic plates. Volcanoes like Eyjafjallajökull captured the world's attention when they erupted, sending ash plumes high into the atmosphere.

        At night, the Icelandic skies became a canvas for the Northern Lights. These ethereal curtains of light danced in shades of green and pink, captivating those who witnessed the celestial display. In remote cabins and cozy cottages, travelers shared stories while sipping on Brennivín, the traditional Icelandic schnapps.

        Iceland was a realm of elemental grandeur, a place where the forces of nature shaped every aspect of life. Your RNN model can use this detailed text to predict the next word in sentences, allowing it to generate vivid and evocative descriptions of Iceland's unique landscapes and the indomitable spirit of its people.

        In the heart of the mystical Sahara Desert, where endless sand dunes met the horizon, life unfolded in the shadow of the eternal sun. The Sahara was a land of vast, undulating landscapes, where the silence was only occasionally interrupted by the whisper of the wind and the cry of desert wildlife.

        Nomadic Tuareg tribes, masters of the desert, traversed the shifting sands on their camels. Their indigo-dyed turbans protected them from the relentless sun, and their intricate knowledge of the desert's secrets guided them through this harsh environment. They carried with them a rich oral tradition of stories and legends.

        Amid the endless expanse of the desert, hidden oasis provided the gift of life. Date palms and other verdant plants offered respite, while ancient villages thrived with centuries-old traditions. Wells were the lifeblood of these settlements, drawing precious water from deep beneath the ground.

        The Sahara was also home to a remarkable diversity of wildlife. Fennec foxes, meerkats, and sand gazelles adapted to the arid conditions, while desert birds like the Egyptian vulture and the hoopoe soared in the skies. At night, the desert was a realm of nocturnal creatures, including the elusive Saharan cheetah.

        As the sun dipped below the sand dunes, the Sahara's colors transformed into a breathtaking display. The sky was painted with vivid oranges and purples, and the stars emerged in all their brilliance. It was a place where the celestial beauty of the night sky was unparalleled.

        The Sahara was a realm of solitude and wonder, where humans and nature coexisted in a delicate balance. Your RNN model can use this descriptive text to predict the next word in sentences, allowing it to generate evocative descriptions of the Sahara Desert and the unique way of life that exists in this arid expanse.

        In the mystical land of Bhutan, nestled high in the Himalayas, life unfolded with a sense of reverence and ancient wisdom. The Bhutanese people, guided by the principles of Gross National Happiness, cherished their culture and the natural world that surrounded them. It was a place where the pursuit of well-being and harmony with the environment took precedence.

        Bhutan was a place of pristine natural beauty. Towering peaks, like Gangkhar Puensum, the world's highest unclimbed mountain, loomed over deep valleys. The forests were home to rare creatures like the red panda and snow leopard. Bhutan's lush terraced fields produced vibrant, organic crops, and prayer flags fluttered in the mountain breezes.

        Local festivals, known as tshechus, celebrated the Bhutanese way of life. Dancers in colorful masks reenacted ancient legends, and the sound of traditional music filled the air. The architecture, with its intricate woodwork and exquisite paintings, reflected the nation's devotion to its unique cultural heritage.

        Visitors to Bhutan were welcomed with warmth and hospitality, and they embarked on treks along ancient mountain paths. The Jomolhari trek led to remote monasteries and offered breathtaking views of snow-capped peaks. Pilgrims made their way to sacred sites, like the Paro Taktsang Monastery, clinging to cliffs like an eagle's nest.

        At night, Bhutan was illuminated by the soft glow of butter lamps in temples and monasteries. The air was filled with the scent of incense, and the echoes of chants and prayers resonated. The skies displayed constellations that had guided Bhutanese navigators for centuries.

        Bhutan was a realm of spiritual harmony and environmental stewardship, where ancient traditions and a focus on happiness coexisted. Your RNN model can use this text to predict the next word in sentences, allowing it to generate evocative descriptions of Bhutan and the unique culture and environment of this Himalayan kingdom.

"""
        ]

# tokenize the words 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
total_words = len(tokenizer.word_index) + 1

# convert sentences to sequences
input_sequences = []
for line in text:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    input_sequences.append(token_list[:i+1])


# Padding Sequences
max_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding = 'pre')


# separate features and classes 
x, y = input_sequences[:,:-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Building and training the neural network with LSTM Model 

model = Sequential([
    Embedding(total_words, 100, input_length=max_length-1),
    LSTM(150, return_sequences = True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics =['accuracy'])

# Train Model
model.fit(x, y, epochs=10, verbose=1)


# save the model
model.save("model.h5")

# import pickle
import pickle

with open("tokenizer.pkl", "wb") as handle:
  pickle.dump(tokenizer, handle)
