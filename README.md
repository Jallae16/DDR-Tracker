# Sigma Style Check

## Inspiration
We love the cool and popular dance moves from Fortnite and pop culture, and we wanted to mix these dance moves with the world of neuro-networks.

## What it does
The project will give the user a random dance that they will perform and they will try to match it up with the model that is dancing on the screen. Then a score will be printed on the screen on how closely they were moving with the model.  We aimed to have our ai model be reached using our custom domain, hosted using cloudflare integration.

## How we built it
We started by collecting a dataset of .mp4 files, using them to train our custom neural network using Mediapipe and Tensorflow. Then, we built a program that would run the network and the game. Unfortunately, confidence in neuro-networks recognition was not high and we moved to a limited problem-type database which allowed us to accomplish the same goal, with a smaller database, and faster speed!

## Challenges we ran into
We had trouble finding plentiful, short-form content of humans doing the dance moves. We also had many issues with our neuro-network. Due to our small and incredibly variant dataset, we weren't able to accurately give a score that represented the dance a person did. We also had trouble hosting the model using Cloudflare.

## Accomplishments that we're proud of
We were able to circumnavigate tons of problems we ran into. Such as when we need lots of shortform training data to build our original dataset, we created a Java program using Tenor (a gif website) API to scrape it and load the .mp4 files into a folder. We also weren't afraid to change our design even so close to the deadline, and we still accomplished what we set out to do.

## What we learned
We gained a whole new pair of eyes into the world of machine learning, neuro-networks, and web integration. Although we could not figure it out for the deadline, we gained so much insight and knowledge that will better help us in our classes and the real world.

## What's next for Sigma Style Check
We plan to expand Sigma Style Check into the world of neuro-networks and commit to improving and having better standards for our database.

Link to Slides: https://www.figma.com/slides/Z5Xkd8KhVNOIvklz8ZhOR1/Sigma-Style-Check?t=dHqvbv0IS6GAL0YF-1
