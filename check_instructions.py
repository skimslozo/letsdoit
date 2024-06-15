import json
from scoring.primitive import get_primitive, SpatialPrimitive



with open('/teamspace/studios/this_studio/letsdoit/descriptions_out.json') as f:
    instruction_list = json.load(f)



for instruction_block in instruction_list:
    instructions = instruction_block['instructions']
    for instruction in instructions:
        spatial_ps = instruction['spatial_primitives']
        for sp in spatial_ps:
            primitive_name = sp['primitive']
            primitive = get_primitive(primitive_name)
            if primitive is None:
                print(f'DANGER! The instruction {instruction["desc_id"]} has a non-existing primitive: {primitive_name}')
            else:
                if primitive == SpatialPrimitive.BETWEEN:
                    if not sp.get('reference_object_2'):
                        print(f'DANGER! The instruction {instruction["desc_id"]} has a "between" primitive that is lacking the reference_object_2')

        

