import click

@click.command()
@click.option('--count', default=1, help='Number of files')
@click.argument('--filename', prompt='Your Name', help='The input .ipynb filename in the current working directory.')
def clear_nb_output(count, filename):
	#clear a target notebook
	!jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace my_notebook.ipynb
	count=count+1  #not used
	ps = f'cleared of output: {click.format_filename(filename}'
	click.echo(ps)

if __name__=='__main__':
	clear_nb_output()
	# import argparse
	# parser = argparse.ArgumentParser(description = 'Say hello')
 #    parser.add_argument('name', help='your name, enter it')
 #    args = parser.parse_args()

 #    main(args.name)
	# clear_nb_output(args.count, filename)