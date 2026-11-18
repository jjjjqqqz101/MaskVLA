"""Bounded, natural-language task contracts. No simulator instance identities."""
from dataclasses import dataclass
import math

from perception_mask.core import validate_box


class StopRequested(RuntimeError):
    pass


def text(value, name, maximum=1500):
    if not isinstance(value,str) or not value.strip() or len(value)>maximum:
        raise ValueError(f'{name} must be nonempty text, at most {maximum} characters')
    return value.strip()


def validate_plan(data, max_steps=4):
    if not isinstance(data,dict):raise ValueError('Plan must be a JSON object')
    summary=text(data.get('summary'),'summary')
    rule=text(data.get('selection_rule'),'selection_rule')
    clarification=data.get('needs_clarification',False)
    if type(clarification) is not bool:raise ValueError('needs_clarification must be boolean')
    rows=data.get('steps')
    if not isinstance(rows,list) or len(rows)>max_steps:raise ValueError(f'The collection area supports at most {max_steps} subtasks')
    steps=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid plan step')
        if row.get('skill','pick_place')!='pick_place':raise ValueError('Only pick_place is available')
        steps.append(dict(skill='pick_place',description=text(row.get('description'),'description',500),
                          destination='green_collection_area',reason=text(row.get('reason'),'reason',600)))
    if len({s['description'] for s in steps})!=len(steps):raise ValueError('Plan repeats a target description')
    return dict(summary=summary,selection_rule=rule,needs_clarification=clarification,steps=steps)


def validate_target(data, width, height):
    if not isinstance(data,dict) or data.get('decision') not in ('pick','complete','uncertain'):
        raise ValueError('Target decision must be pick/complete/uncertain')
    result=dict(decision=data['decision'],reason=text(data.get('reason'),'reason',1000))
    if data['decision']=='pick':
        result.update(description=text(data.get('description'),'target description',500),
            bbox_xyxy=validate_box(data.get('bbox_xyxy'),width,height,
                                  data.get('coordinate_space','normalized_1000')))
    return result


def validate_judgement(data):
    if not isinstance(data,dict):raise ValueError('Judgement must be a JSON object')
    if data.get('assessment') not in ('not_grasped','grasped_lifted','placed','failed','uncertain'):
        raise ValueError('Unknown visual assessment')
    if data.get('decision') not in ('continue','retry','abort','next','complete'):
        raise ValueError('Unknown visual decision')
    confidence=data.get('confidence',0)
    if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:
        raise ValueError('Visual confidence must be finite within 0..1')
    return dict(assessment=data['assessment'],decision=data['decision'],confidence=float(confidence),
                reason=text(data.get('reason'),'visual evidence',1200))


def validate_completion(data):
    if not isinstance(data,dict) or type(data.get('complete')) is not bool:
        raise ValueError('Final visual judgement must explicitly return complete=true/false')
    return dict(complete=data['complete'],reason=text(data.get('reason'),'completion evidence',1200),
                remaining_description=str(data.get('remaining_description',''))[:1200])


def validate_mask_check(data):
    if not isinstance(data,dict) or type(data.get('valid')) is not bool:
        raise ValueError('Mask confirmation must explicitly return valid=true/false')
    return dict(valid=data['valid'],reason=text(data.get('reason'),'mask visual evidence',1200))
