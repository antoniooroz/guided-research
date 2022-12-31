from run import run
import  pgnn.utils.arguments_parsing as arguments_parsing

args = arguments_parsing.parse_args()
run(config=args.config)