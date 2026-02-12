from utils.register import registry
from utils.options import get_parser
import engine
import agents
import scenario
import utils
import os
import asyncio


if __name__ == '__main__':
    args = get_parser()
    scenario = registry.get_class(args.scenario)(args)
    asyncio.run(scenario.parallel_run())  
    # scenario.run()