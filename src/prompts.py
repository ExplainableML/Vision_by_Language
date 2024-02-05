###################################
###### GENERAL PROMPTS ############
###################################
short_modifier_prompt = """
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". Each time generate one instruction and one edited description only."""

short_focus_object_modifier_prompt = """
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The instruction contains an object occuring in the image. The edited description you generate should begin with "Edited Description".
The edited description should list the instruction object first, then any other object and scene from the image content. 
All attributes that do not belong to the instruction object should be removed. The edited description should be a comma-separated list.
Each time generate one edited description only. Use these three examples as reference:

Image Content: There is a dining room with blue chairs and a red refrigerator.
Instruction: refrigerator
Edited Description: red refrigerator, chairs, dining room.

Image Content: A white laptop on a yellow table in a dark room.
Instruction: laptop
Edited Description: white laptop, table, dark room.

Image Content: A long yellow train on old and dirty tracks underneat a gray sky.
Instruction: tracks
Edited Description: old and dirty tracks, train, gray sky.
"""

simple_modifier_prompt = """
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The edited description you generate  should be complete and can cover various semantic aspects, 
including cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. 
The edited description needs to be as simple as possible. The instruction does not need to explicitly indicate which
type it is. Avoid adding imaginary things. Each time generate one instruction and one edited description only. Keep the edited description as short as possible"""


contextual_modifier_prompt = '''
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The caption you generate should only talk about the modified image and not the original image. 
Keep the caption as short as possible. Avoid adding imaginary things. Use the examples below for reference.

Image Content: the men are holding a large fish on a stick
Instruction: People watch behind the fences of animals in the center.
Edited Description: people, a bull, a crowd, a fence, a ring, a bull, a ring, a

Image Content: a man in a blue robe is adjusting a man in a blue robe
Instruction: Change the gowns to black with a navy blue collar
Edited Description: a woman in a cap and gown is standing in front of a group of people

Image Content: people, sled dogs, snow, sled, sled dogs, sled, sled dogs, sled
Instruction: Dog led sled moves in front of another person behind it in the snow.
Edited Description: a man is pulling a sled full of huskies down a snowy trail

Image Content: the penguin, the snow, the penguin, the snow, the penguin, the penguin, the penguin, the penguin,
Instruction: The Target Image shows a single penguin standing on the ice with a fish in its beak.
Edited Description: the penguin is holding a fish in its beak

Image Content: the panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the walls with wooden fences.
Edited Description: Three dogs are laying down on a deck

Image Content: The dog is playing with a dachshund on the beach
Instruction: Remove the small dog and have the large dog face the opposite direction.
Edited Description: The dog is standing in the sand on a beach

Image Content: there is a bottle of pepsi cola sitting on a table
Instruction: show three bottles of soft drink
Edited Description: Three bottles of soft drink sitting on a table.

Image Content: a group of people posing for a photo in a lab
Instruction: reduce the number of people to four, make them stand outside on grass
Edited Description: Four people posing for a photo standing on grass.

Image Content: There is a herd of antelopes on a dirt road
Instruction: The target photo has one antelope in a wooded area looking back at the camera.
Edited Description: One antelope in a wooded area looking back at the camera.

Image Content: There is a bakery with a counter full of baked goods
Instruction: The target photo is of a sticker display in a store.
Edited Description: A sticker display in a store.

Image Content: The doctor is holding up a syringe
Instruction: Change to a close-up photograph of a female nurse and a real syringe, must be looking directly towards camera
Edited Description: A close-up of a female nurse holding a syringe, looking directly towards the camera.

Image Content: Two dogs are sitting in the back of a truck
Instruction: Push the brown dog on the blue float in the pool.
Edited Description: The brown dog is on a blue float in the pool.

Image Content: The baboon is sitting in a tree
Instruction: make monkey stand on top of a bucket
Edited Description: The baboon is standing on top of a bucket.

Image Content: three people are swimming with sting rays
Instruction: Unlike the Reference Image, I want the Target Image to show a single person snorkeling and holding a sea turtle
Edited Description: A single person snorkeling and holding a sea turtle.

Image Content: There is a plate with lemons, onions, and fish on it
Instruction: change shrimps to bowls of fruits
Edited Description: There is a plate with lemons, onions, and bowls of fruits on it.

Image Content: There is a glass bowl filled with a brown mixture on a wooden table
Instruction: The target photo shows the rising of a ball of dough at the top of the same white bowl
Edited Description: A ball of dough rising at the top of a white bowl on a wooden table.

Image Content: There is a silver water bottle on a black background
Instruction: Remove thermal bottle, Add pump-style bottle (goldtone pump, white cylindrical bottle with marble finish), Change to light background
Edited Description: A pump-style bottle with a goldtone pump and a white cylindrical bottle with a marble finish on a light background.

Image Content: There is a bottle of beer next to a glass of beer
Instruction: Put the opened green bottle in front of woman on the phone. 
Edited Description: An opened green bottle of beer in front of a woman on the phone.

Image Content: a group of boys are posing on a bed with pillows
Instruction: Change to a bigger pile of pillows, cushions with variant colours, no people in view
Edited Description: A bigger pile of pillows and cushions with variant colors, no people in view.

Image Content: There is a black and brown puppy sitting next to a teddy bear
Instruction: Add an owner with the dog on its leash
Edited Description: A black and brown puppy on a leash next to its owner and a teddy bear.

Image Content: a pug dog is running in a grassy field
Instruction: Add a human wearing a pug mask
Edited Description: A pug dog and a human wearing a pug mask are running in a grassy field.

Image Content: the dog is wearing a mustache and a bowtie
Instruction: Have two different dogs on the ground facing left but one standing on its hind legs.
Edited Description: Two different dogs on the ground facing left, one standing on its hind legs, both wearing a mustache and a bowtie.

Image Content: The image shows a group of pins on a piece of paper
Instruction: Add spheres to the end of the safety pins
Edited Description: A group of pins with spheres on the ends on a piece of paper.

Image Content: the gorilla is eating a banana from a bag
Instruction: Facing the other direction, closer photo
Edited Description: A closer photo of the gorilla facing the opposite direction, eating a banana from a bag.
'''



structural_modifier_prompt = '''
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The edited description you generate should be complete and can cover various semantic aspects, such as 
cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:

"cardinality"
Image Content: A bee is flying around a flower on a field.
Instruction: Duplicate the flower the bee is flying around.
Edited Description: A bee is flying around two flowers on a field.

"addition"
Image Content: A dog is walking in a grassy field.
Instruction: Add a butterfly to the scene.
Edited Description: A dog is walking next to a butterfly in a grassy field.

"negation"
Image Content: A plane flying in the cloudy sky.
Instruction: Remove the clouds from the sky.
Edited Description: A plane flying in the clear sky.

"direct addressing"
Image Content: The eiffel tower on a summer day.
Instruction: Highlight the eiffel tower with a red circle.
Edited Description: The eiffel tower, highlighted with a red circle, on a summer day.

"compare & change"
Image Content: The panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the bamboo with bones.
Edited Description: Three dogs are laying in the grass munching on some bones.

"comparative"
Image Content: A picture of the sun and the moon.
Instruction: Make the sun brighter than the moon.
Edited Description: A picture of the sun shining much brighter than the moon.

"conjunction"
Image Content: Several children are playing on the ground.
Instruction: Add both a cat and a dog to the image.
Edited Description: Several children are playing with a cat and a dog.

"spatial relations & background"
Image Content: A man standing on a sled pulled by several sled dogs, next to a small wooden house.
Instruction: Place the house behind the man, and make it much larger.
Edited Description: A man standing on a sled pulled by several sled dogs, in front of a large wooden house.

"viewpoint"
Image Content: A picture of a salad bowl.
Instruction: Change the perspective to a bird's-eye view.
Edited Description: A salad bowl viewed from above. 
 

The edited description needs to be as simple as possible. The instruction does not need to explicitly indicate which
type it is. Avoid adding imaginary things. Each time generate one instruction and one edited description only. Keep the edited description as short as possible. Here are some more examples for reference:

Image Content: the men are holding a large fish on a stick
Instruction: People watch behind the fences of animals in the center.
Edited Description: people, a bull, a crowd, a fence, a ring, a bull, a ring, a

Image Content: a man in a blue robe is adjusting a man in a blue robe
Instruction: Change the gowns to black with a navy blue collar
Edited Description: a woman in a cap and gown is standing in front of a group of people

Image Content: people, sled dogs, snow, sled, sled dogs, sled, sled dogs, sled
Instruction: Dog led sled moves in front of another person behind it in the snow.
Edited Description: a man is pulling a sled full of huskies down a snowy trail

Image Content: the legend of zelda ocarina of time t-shirt
Instruction: is green and a graphic on it and is green
Edited Description: the shirt is green with an image of link holding a sword

Image Content: i'm glad you're alive
Instruction: is darker and less wordy and is darker
Edited Description: the shirt is burgundy with a pug face on it

Image Content: the penguin, the snow, the penguin, the snow, the penguin, the penguin, the penguin, the penguin,
Instruction: The Target Image shows a single penguin standing on the ice with a fish in its beak.
Edited Description: the penguin is holding a fish in its beak

Image Content: the panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the walls with wooden fences.
Edited Description: Three dogs are laying down on a deck

Image Content: The dog is playing with a dachshund on the beach
Instruction: Remove the small dog and have the large dog face the opposite direction.
Edited Description: The dog is standing in the sand on a beach

Image Content: the dress is a silver sequined one shoulder dress
Instruction: is lighter colored and less fitted and is a light pink color
Edited Description: the dress is a one shoulder chiffon dress with a ruffled skirt

Image Content: black and yellow hawaiian floral print sleeveless hawaiian hawaiian 
Instruction: Sexier and no vibrant colors and less revealing chest and more evening wear
Edited Description: the dress is a black and green dress with a sleeveless bodice and a flared skirt
'''



#############################
###### BLIP PROMPT ##########
#############################
blip_prompt = 'Describe the image in complete detail. You must especially focus on all the objects in the image'




###################################
###### FASHION-IQ PROMPT ##########
###################################
structural_modifier_prompt_fashion = '''
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The edited description you generate should be complete and can cover various semantic aspects, such as 
cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:

"cardinality"
Image Content: A bee is flying around a flower on a field.
Instruction: Duplicate the flower the bee is flying around.
Edited Description: A bee is flying around two flowers on a field.

"addition"
Image Content: A dog is walking in a grassy field.
Instruction: Add a butterfly to the scene.
Edited Description: A dog is walking next to a butterfly in a grassy field.

"negation"
Image Content: A plane flying in the cloudy sky.
Instruction: Remove the clouds from the sky.
Edited Description: A plane flying in the clear sky.

"direct addressing"
Image Content: The eiffel tower on a summer day.
Instruction: Highlight the eiffel tower with a red circle.
Edited Description: The eiffel tower, highlighted with a red circle, on a summer day.

"compare & change"
Image Content: The panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the bamboo with bones.
Edited Description: Three dogs are laying in the grass munching on some bones.

"comparative"
Image Content: the man is wearing a red t - shirt
Instruction: is solid white and is a ligher color
Edited Description: the man is wearing a solid,  light-white t-shirt.

"conjunction"
Image Content: Several children are playing on the ground.
Instruction: Add both a cat and a dog to the image.
Edited Description: Several children are playing with a cat and a dog.

"spatial relations & background"
Image Content: A man standing on a sled pulled by several sled dogs, next to a small wooden house.
Instruction: Place the house behind the man, and make it much larger.
Edited Description: A man standing on a sled pulled by several sled dogs, in front of a large wooden house.

"viewpoint"
Image Content: A picture of a salad bowl.
Instruction: Change the perspective to a bird's-eye view.
Edited Description: A salad bowl viewed from above. 
 
The edited description needs to be as simple as possible. The instruction does not need to explicitly indicate which
type it is. Avoid adding imaginary things. Each time generate one instruction and one edited description only. Keep the edited description as short as possible. Here are some more examples for reference:

Image Content: the man is wearing a white tank top and black shorts
Instruction: looks faded and cheaper and is longer
Edited Description: The man is wearing a faded, cheap-looking and elongated white tank top, giving a worn-out slightly oversized and more casual appearance.

Image Content: the only winning move is not to play t shirt
Instruction: The shirt is black with a skeleton and is red
Edited Description: The image shows a black t-shirt with a red skeleton design on it. Is says "the only winning move is not to play".

Image Content: the man is wearing a black polo shirt
Instruction: is less formal with less buttons and is gray with no collar
Edited Description: The man is wearing a less formal, gray polo shirt with no collar and fewer buttions, giving it a more casual and relaxed appearance.

Image Content: the legend of zelda ocarina of time t-shirt
Instruction: is green and a graphic on it and is green
Edited Description: the shirt is green with an image of link holding a sword

Image Content: The woman is wearing a tan shirt and jeans
Instruction: is the same and appears to be exactly the same
Edited Description: The woman is wearing a tan shirt and jeans

Image Content: The woman is wearing a green top with hearts on it
Instruction: is pinched more below the bust and is brown in color
Edited Description: The woman is wearing a brown top with hearts on it, which is pinched more below the bust.

Image Content: the woman is wearing a yellow tank top
Instruction: is very similar but deeper red and is red
Edited Description: The woman is wearing a deep red tank top.

Image Content: the woman is wearing a polka dot dress
Instruction: has thinner straps and is darker
Edited Description: The woman is wearing a darker polka dot dress with thin straps, giving it a more delicate, elegant look.

Image Content: the dress is a silver sequined one shoulder dress
Instruction: is lighter colored and less fitted and is a light pink color
Edited Description: the dress is a one shoulder chiffon dress with a ruffled skirt

Image Content: black and yellow hawaiian floral print sleeveless hawaiian hawaiian 
Instruction: Sexier and no vibrant colors and less revealing chest and more evening wear
Edited Description: the dress is a black and green dress with a sleeveless bodice and a flared skirt
'''
