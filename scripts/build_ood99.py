"""Build explicit task-specific AP specifications from the installed official BDDL."""
import hashlib
import json
import re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Every target is reviewed against the task's objects. Prefer distinctive rigid
# objects/tools; never use particle systems, future sliced objects, or floors.
TARGETS = '''assembling_gift_baskets wicker_basket
attach_a_camera_to_a_tripod digital_camera
boxing_books_up_for_storage box
boxing_food_after_dinner tupperware
bringing_in_wood plywood
bringing_paper_to_recycling sack
bringing_water bottle
can_meat hinged_jar
canning_food carving_knife
carrying_in_groceries sack
carrying_out_garden_furniture barrow
chop_an_onion parer
chopping_wood ax
clean_a_keyboard pipe_cleaner
clean_a_patio broom
clean_a_trumpet cornet
clean_boxing_gloves boxing_glove
clean_up_broken_glass ashcan
clean_up_your_desk stapler
clean_your_rusty_garden_tools scraper
cleaning_up_branches_and_twigs recycling_bin
cleaning_up_plates_and_food bowl
clearing_food_from_table_into_fridge tupperware
collecting_aluminum_cans bucket
collecting_childrens_toys teddy
composting_waste half__banana
cook_a_brisket brisket
cook_a_frozen_pie apple_pie
cook_bacon bacon
cook_broccolini frying_pan
cook_brussels_sprouts carving_knife
cook_cabbage head_cabbage
cook_hot_dogs hotdog
dispose_of_batteries battery
dispose_of_glass water_glass
freeze_fruit tupperware
freeze_pies apple_pie
getting_organized_for_work mouse
halve_an_egg carving_knife
hanging_pictures poster
hiding_Easter_eggs easter_egg
installing_a_fax_machine facsimile
installing_a_modem modem
installing_a_scanner scanner
installing_smoke_detectors fire_alarm
laying_tile_floors tile
loading_the_car digital_camera
make_cabinet_doors cabinet_door
make_gift_bags_for_baby_showers sack
make_microwave_popcorn popcorn__bag
make_pizza cookie_sheet
make_rose_centerpieces rose
moving_boxes_to_storage storage_container
organizing_art_supplies glue_stick
organizing_school_stuff calculator
outfit_a_basic_toolbox drill
packing_meal_for_delivery wrapped_hamburger
picking_up_toys tennis_ball
picking_up_trash can__of__soda
polishing_shoes scrub_brush
preparing_lunch_box packing_box
put_together_a_basic_pruning_kit pruner
putting_away_Halloween_decorations candle
putting_away_toys toy_figure
putting_dirty_dishes_in_sink bowl
putting_dishes_away_after_cleaning plate
putting_shoes_on_rack gym_shoe
putting_up_Christmas_decorations_inside gift_box
re_shelving_library_books book
rearrange_your_room tissue_dispenser
rearranging_kitchen_furniture toaster
scrubbing_bathroom_floor scrub_brush
set_up_a_coffee_station_in_your_kitchen paper_coffee_filter
setting_mousetraps mousetrap
setting_the_fire cigar_lighter
setting_the_table tablefork
setup_a_bar_for_a_cocktail_party corkscrew
slicing_vegetables parer
sorting_books_on_shelf hardback
sorting_bottles_cans_and_paper wine_bottle
sorting_household_items toothbrush
sorting_vegetables mixing_bowl
spraying_for_bugs insectifuge__atomizer
spraying_fruit_trees pesticide__atomizer
stacking_wood log
store_batteries battery
store_honey jar__of__honey
store_produce mango
storing_food box__of__oatmeal
sweeping_garage broom
thawing_frozen_food chicken_breast
tidying_bathroom soap_dish
tidying_bedroom sandal
tidying_living_room newspaper
turning_out_all_lights_before_sleep table_lamp
unloading_the_car briefcase
vacuuming_floors vacuum
wash_a_baseball_cap baseball_cap
wash_dog_toys teddy'''

def main():
    definitions = Path('/root/BEHAVIOR-1K-complete/bddl3/bddl/activity_definitions')
    metadata = Path('/root/download/datasets/BEHAVIOR-1K/2026-challenge-task-instances/metadata/available_tasks.yaml')
    tasks = yaml.safe_load(metadata.read_text())
    targets = dict(line.split() for line in TARGETS.splitlines())
    override_path=ROOT/'configs/ood99_overrides.json'
    overrides=json.loads(override_path.read_text()) if override_path.exists() else {}
    assert set(targets) == set(tasks) - {'turning_on_radio'} and len(targets) == 99
    specs = []
    for index, (task, prefix) in enumerate(targets.items()):
        source = definitions / task / 'problem0.bddl'
        text = source.read_text()
        objects = text.split('(:objects')[1].split('(:init')[0]
        keys = list(dict.fromkeys(re.findall(r'\b(' + re.escape(prefix) + r'\.n\.\d+_\d+)\b', objects)))
        assert keys, (task, prefix)
        init = text.split('(:init')[1].split('(:goal')[0]
        # Only positive top-level init predicates; never sample a negated relation.
        stack = [[]]
        for token in re.findall(r'\(|\)|[^\s()]+', '(' + init):
            if token == '(':
                child = []; stack[-1].append(child); stack.append(child)
            elif token == ')':
                if len(stack) > 1: stack.pop()
            else: stack[-1].append(token)
        relations = [tuple(p) for p in stack[0][0] if isinstance(p, list) and len(p) == 3 and p[0] in ('ontop','inside')]
        by_child = {child: (rel, parent) for rel, child, parent in relations}
        placements = []
        for key in keys:
            rel, parent = by_child.get(key, ('unresolved', None))
            chain = []
            current = key
            for _ in range(8):
                if current not in by_child: break
                relation, ancestor = by_child[current]
                chain.append({'relation': relation, 'object_key': ancestor})
                current = ancestor
            placements.append({'key': key, 'relation': rel, 'support_key': parent, 'ancestors': chain})
        spec = {
            'task_name': task, 'instance_id': 301, 'seed': 20261019 + index,
            'target_key': keys[0], 'target_keys': keys,
            'target_description': prefix.replace('__', ' ').replace('_', ' '),
            'placements': placements,
            'condition': 'relocate_on_original_support_and_close_ancestor_if_available',
            'ood_axis': 'new_object_positions_plus_initial_search_heading',
            'minimum_relocation_m': 0.08, 'placement_attempts': 12,
            'heading': 'maximize_angular_separation_from_all_targets_after_evaluator_reset',
            'goal': '寻找并接近' + prefix.replace('__', ' ').replace('_', ' ') + '；有多个同类物体时找到其中任意一个即可。必要时搜索可见通路并检查封闭容器。',
            'acceptance': {'max_horizontal_distance_m': 1.0, 'min_target_pixels': 16,
                           'initial_all_targets_invisible': True, 'planner_reached_required': True,
                           'original_task_completion_required': False},
            'status': 'designed_not_simulator_validated',
            'definition_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'semantic_policy': 'Preserve original support/containment and object attributes. No arbitrary container insertion. Same-class targets are all perturbed and any one may satisfy the goal.',
            'limitations': ['Scene perturbation OOD only; not proof of statistical separation from all training layouts.',
                           'One public-test instance and one seed per task; not a robust task success-rate estimate.'],
        }
        spec.update(overrides.get(task,{}))
        (ROOT / 'configs/ood99' / (task + '.json')).write_text(json.dumps(spec, ensure_ascii=False, indent=2))
        specs.append(spec)
    (ROOT / 'configs/ood99_manifest.json').write_text(json.dumps(specs, ensure_ascii=False, indent=2))
    print(json.dumps({'tasks': len(specs), 'unresolved': [s['task_name'] for s in specs if any(p['relation']=='unresolved' for p in s['placements'])]}))

if __name__ == '__main__': main()
