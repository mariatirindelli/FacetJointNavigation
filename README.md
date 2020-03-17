# FacetJointNavigation

This is an implementation of the image processing pipeline described in https://arxiv.org/abs/2002.11404

A python server (Currently only OpenIgtLink protocol is implemented) waits for a client connection and then starts receiving 
ultrasound images and force data. Such data are pre-processed and analyzed through a 2D + 1D Convolutional Network to detect
verterbae location along the spine. 

Once the correct vertebra location is detected (the desired vertebral label is reported in the initial config file), the vertebra
location is sent to the client. 

Currently, the server-client communication is only implemented with OpenIgtLink protocol. The OpenIgtlink server receive images
as OpenIgtlink Image Message and force data as OpenIgtlink Force Message. OpenIgtLink Status Messages are used to communicate 
the client status and the kind of processing to be performed on the images. 
The vertebral location is sent to the client as a OpenIgtlink Position message. 
