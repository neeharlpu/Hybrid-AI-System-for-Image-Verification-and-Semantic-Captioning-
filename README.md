# Hybrid AI for Semantic Captioning and Image Verification

## Project Overview

Hybrid AI for Semantic Captioning and Image Verification is an AI-based image analysis system that combines image authenticity verification with semantic image captioning.

The system is designed to analyze an uploaded image from two perspectives:

1. Image Verification — determine whether the image is real or AI-generated.
2. Semantic Captioning — generate a meaningful textual description of the image.

The project also includes a multilingual component, with Malayalam translation being explored using IndicTrans2.

The main objective is to bring image verification, image understanding, explainable AI, and multilingual natural language processing together in a single application.

---

## How the System Works

The overall workflow is:

```text
                         Input Image
                              |
                              v
                    Image Preprocessing
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Image Verification        Semantic Captioning
          Real / AI-Generated              BLIP
                 |                         |
                 v                         v
            Verification              Generated Caption
              Result                       |
                 |                         v
                 v                    Translation
             Grad-CAM++                    |
            Explanation                    v
                                      Multilingual
                                         Caption

The verification and captioning components perform different tasks but work together as part of the same image analysis pipeline.

Image Verification

The image verification component classifies an image into two categories:

Real
AI-generated

The project has explored a ResNet18-based classifier for this task.

The verification module does not only provide a classification result. Grad-CAM++ is used to generate visual explanations that show the regions of the image that contributed to the model's prediction.

The verification workflow is:

Image
  |
  v
Preprocessing
  |
  v
Verification Model
  |
  v
Real / AI-Generated
  |
  v
Grad-CAM++
  |
  v
Visual Explanation
Semantic Image Captioning

The captioning component generates a textual description of the uploaded image.

BLIP is used for image captioning in the current approach.

The captioning process can be represented as:

Input Image
     |
     v
Visual Understanding
     |
     v
BLIP
     |
     v
Generated Caption

Earlier experiments also explored an InceptionV3 and LSTM-based image captioning approach using Flickr image-caption datasets.

These experiments helped in understanding the image-to-text captioning pipeline.

Multilingual Captioning

The system can extend the generated English caption into a multilingual output.

IndicTrans2 was explored for translation, particularly for Malayalam.

The intended workflow is:

Image
  |
  v
Generated English Caption
  |
  v
Translation
  |
  v
Malayalam / Other Supported Languages

The multilingual component is intended to make the generated image descriptions accessible to users beyond English.

Explainable AI

Explainability is an important part of the image verification component.

Instead of showing only a prediction such as:

Prediction: AI-Generated

the system can provide a visual explanation using Grad-CAM++.

The explanation can include:

Original image
Verification result
Confidence
Grad-CAM++ heatmap
Highlighted regions that influenced the prediction

This helps provide more insight into the model's decision instead of treating the verification model as a black box.

Captioning Experiments
Flickr8k

Flickr8k was used during the earlier image captioning experiments.

The implementation included:

InceptionV3 for image feature extraction
Tokenized captions
LSTM-based caption generation

The recorded setup had approximately 8,811 vocabulary tokens and a maximum caption length of approximately 38.

Flickr30k

Flickr30k was also explored during the captioning experiments.

The dataset used in the experiments contained approximately 31,785 images, with a vocabulary of approximately 19,978 and a maximum caption length of approximately 83.

These experiments were part of the development and evaluation of the image captioning pipeline.

System Architecture

The system can be divided into four major layers.

1. Input Layer

The user uploads an image through the application.

2. AI Processing Layer

The uploaded image is processed by the verification and captioning models.

The verification pipeline performs:

Image
  |
  v
Preprocessing
  |
  v
Verification Model
  |
  v
Real / AI-Generated

The captioning pipeline performs:

Image
  |
  v
Image Understanding
  |
  v
BLIP
  |
  v
Caption
3. Explainability and Language Layer

The verification result can be visualized using Grad-CAM++.

The generated caption can be translated into supported languages.

4. Application Layer

The components are brought together through a Streamlit-based application interface.

Technologies Used
Programming
Python
Deep Learning
PyTorch
Transformers
ResNet18
BLIP
InceptionV3
LSTM
Computer Vision
OpenCV
Grad-CAM++
Natural Language Processing
Image Captioning
IndicTrans2
Multilingual Translation
Application Development
Streamlit
Project Structure

The repository can be organized around the following components:

Hybrid-AI-Semantic-Captioning-and-Image-Verification/
│
├── models/
│   ├── verification/
│   └── captioning/
│
├── data/
│
├── notebooks/
│
├── src/
│   ├── verification/
│   ├── captioning/
│   ├── translation/
│   └── explainability/
│
├── app/
│
├── requirements.txt
├── README.md
└── ...

The actual repository structure may vary depending on the final implementation.

Core Concept

The main idea of the project is to combine multiple AI capabilities instead of treating image verification and image captioning as separate applications.

                         IMAGE
                           |
                +----------+----------+
                |                     |
                v                     v
          Verification            Captioning
                |                     |
                v                     v
         Real / AI-Fake          Description
                |                     |
                v                     v
            Grad-CAM++           Translation
                |                     |
                +----------+----------+
                           |
                           v
                      Final Output

The system therefore aims to answer two important questions:

Is this image real or AI-generated?

and

What is happening in this image?

Project Objective

The objective of this project is to develop a hybrid AI system that combines:

Computer Vision + Image Verification + Semantic Captioning + Explainable AI + Multilingual NLP

into a single application for image analysis and understanding.

Current Scope:

The project currently focuses on:

-Image authenticity verification
-Semantic image caption generation
-Explainable AI for verification
-Multilingual caption translation
-Streamlit-based application development

The project is being developed as part of an MSc Data Science project.

Future Extensions

Possible future improvements include:

-Improving image verification performance
-Supporting additional languages
-Improving caption generation quality
-Providing more detailed model explanations
-Exploring stronger image-generation detection models
-Improving the overall application interface and inference pipeline

These are considered future extensions and are not presented as currently completed features.

Summary

Hybrid AI for Semantic Captioning and Image Verification combines image authenticity analysis and semantic image understanding into a single AI workflow.

The system takes an image as input and provides an authenticity prediction, an automatically generated description, an explainable visualization for the verification result, and multilingual captioning capabilities.

The project brings together computer vision, deep learning, explainable AI, image captioning, and multilingual natural language processing to create a unified image analysis system.


