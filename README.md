# Apparel-Attributes-Recognition

Here I make use of Google's Inception v3 model to recognize attributes of apparels. Attributes include colour, type of collar, sleeve length and pattern among others. 

## Data Set

The dataset used to train the model is the [Clothing Attribute Dataset](https://purl.stanford.edu/tb980qz1002). It consists of around 1856 upper body images. Each image is assigned to particular classes indicating the colour, sleeve length, collar type and others. For example, a T-Shirt might be assigned 1 if its black in colour or 0 if it's not black and similar annotation for other attributes. <br/>
Sample Image and attributes from the dataset.
<div>
<img src="https://raw.githubusercontent.com/esha-sg/Apparel-Attributes-Recognition/master/000008.jpg"
width="230"height="290">
Full Sleeve ; Black colour ; Woman ; Low Skin Exposure ....
<br/>
</div>

## Training and Model

The Inception v3 model is employed to get the feature vectors of the images which are 2048 dimensional vector. Once these values are extracted the last layer of the Inception v3 is replaced with 26 fully connected layers to recognize individual attribute. The fully connected layers make use of ReLU activation function. Each layer is trained for 80 epochs.

## Results

Following predictions were made on an image from the validation set.<br/>
<div>
<img src="https://raw.githubusercontent.com/esha-sg/Apparel-Attributes-Recognition/master/validation.jpg"
width="230"height="290">
Half Sleeves ; Black colour ; Man ; Low Skin Exposure ...
<br/>
</div>


