import os
from nanovvla import VLA #, SamplingParams
# import tokenizers?

POLICY_PATH  = "grahamwichhh/v3_smolvla_so101-pick-up-lego"

# directly mirroring nano-vllm
def main():

    # setup path to HF model
    path = os.path.expanduser(POLICY_PATH)
    
    # tokenize? (should happen inside model)

    # wrapper class:
    vla = VLA(path) # enforce_eager=True, tensor_parallel_size=1)

    # sampling params? for vllm, this is like temparature

    # prompts?
    # NOTE: as of now, the VLA usage appears to be 1 policy per 1 prompt
    #   ex: dataset=pi0 where prompt is "pick up lego", means no other 
    #       prompt will interact with this trained vla policy *as well*
    #       (research has shown similar prompts work ok)
    # TODO: this should not be like this, but there doesnt seem to be code
    #       for this. would only need a VLM logic abstraction layer in-between?

    # OPTION1: get direct output tensors from the model that can be sent
    #   directly to a robot action middleware like ROS2 or Lerobot
    # outputs = vla.generate()
    # single_output = vla.generate_single_step() # eval mode, decoupled from hardware

    # OPTION2: run a parameterized VLA loop that will attempt to run with 
    #   some framework that can be sent to the model:
    # vla.run(FRAMEWORK, NUM_SECONDS)



    



if __name__ == "__main__":
    main()
    