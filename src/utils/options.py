import argparse
from utils.register import registry
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario', 
        default='J1Bench.Scenario.CR',
        choices = [
            'J1Bench.Scenario.KQ',
            'J1Bench.Scenario.LC',
            'J1Bench.Scenario.CD',
            'J1Bench.Scenario.DD',
            'J1Bench.Scenario.CI',
            'J1Bench.Scenario.CR'
        ],
        type=str
        )
    
    args, _ = parser.parse_known_args()
    
    scenario_group = parser.add_argument_group(
        title='Scenario',
        description='Scenario configuration'
    )
    
    # 不同的场景需要不同的参数
    registry.get_class(args.scenario).add_parser_args(scenario_group)
    args, _ = parser.parse_known_args()
    
    if hasattr(args, "general_public"):
        general_public_group = parser.add_argument_group(
            title='general_public',
            description='general public configuration'
        )
        
        if registry.get_class(args.general_public) is not None:
            registry.get_class(args.general_public).add_parser_args(general_public_group)
        else:
            raise ValueError('general public is not defined in the scenario')

    if hasattr(args, "specific_character"):
        specific_character_group = parser.add_argument_group(
            title='specific_character',
            description='specific character configuration'
        )
        
        if registry.get_class(args.specific_character) is not None:
            registry.get_class(args.specific_character).add_parser_args(specific_character_group)
        else:
            raise ValueError('specific character is not defined in the scenario')
        
    if hasattr(args, "trainee"):
        trainee_group = parser.add_argument_group(
            title="trainee",
            description="trainee configuration",
        )
        if registry.get_class(args.trainee) is not None:
            registry.get_class(args.trainee).add_parser_args(trainee_group)
        else:
            raise RuntimeError()
    
    if hasattr(args, "lawyer"):
        lawyer_group = parser.add_argument_group(
            title="Lawyer",
            description="Lawyer configuration",
        )
        if registry.get_class(args.lawyer) is not None:
            registry.get_class(args.lawyer).add_parser_args(lawyer_group)
        else:
            raise RuntimeError()
    
    if hasattr(args, "plaintiff"):
        plaintiff_group = parser.add_argument_group(
            title="plaintiff",
            description="plaintiff lawyer configuration",
        )
        if registry.get_class(args.plaintiff) is not None:
            registry.get_class(args.plaintiff).add_parser_args(plaintiff_group)
        else:
            raise RuntimeError()
        
    if hasattr(args, "defendant"):
        defendant_group = parser.add_argument_group(
            title="defendant",
            description="defendant lawyer configuration",
        )
        if registry.get_class(args.defendant) is not None:
            registry.get_class(args.defendant).add_parser_args(defendant_group)
        else:
            raise RuntimeError()
        
    if hasattr(args, "judge"):
        judge_group = parser.add_argument_group(
            title="judge",
            description="judge configuration",
        )
        if registry.get_class(args.judge) is not None:
            registry.get_class(args.judge).add_parser_args(judge_group)
        else:
            raise RuntimeError()
        
    if hasattr(args, "procurator"):
        procurator_group = parser.add_argument_group(
            title="procurator",
            description="procurator configuration",
        )
        if registry.get_class(args.procurator) is not None:
            registry.get_class(args.procurator).add_parser_args(procurator_group)
        else:
            raise RuntimeError()
    
    args, _ = parser.parse_known_args()
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")

    return args
