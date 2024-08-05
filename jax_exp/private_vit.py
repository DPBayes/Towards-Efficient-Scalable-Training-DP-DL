import flax.linen as nn
from transformers import FlaxViTModel


#Load pretrained ViT model and add a dense layer on top as the classifier
class ViTModelHead(nn.Module):
    #config: FlaxViTModel.config_class #ViT configuration. Architecture
    num_classes: int #Num classes, or output neurons
    vit: nn.Module
    pretrained_model: FlaxViTModel

    @nn.compact
    def __call__(self, inputs):
        #inputs = processor.preprocess(images=inputs, return_tensors="np")
        #after_inputs = self.vit(inputs).pooler_output
        #inputs = self.vit(inputs).pooler_output

        outputs = self.pretrained_model(inputs)
        x = outputs.last_hidden_state[:,0]
        #The result of the vit module is a FlaxBaseModelOutputWithPooling object
        #Which is a tuple of two arrays. The first one is a last_hidden_state array with size (1,197,768)
        #The second one is a pooler_output array, with size (1,768)
        #So, I take the results of the pooler output and give them to the last classifier layer
        #hidden_states = after_inputs.pooler_output
        #print('hidden_states shape',hidden_states.shape)
        #inputs = nn.Dense(self.num_classes,name='classifier',kernel_init=nn.zeros)(inputs)

        x = nn.Dense(features=self.num_classes,name='classifier',kernel_init=nn.zeros)(x)
    
        #logits = self.classifier(hidden_states)
        return x