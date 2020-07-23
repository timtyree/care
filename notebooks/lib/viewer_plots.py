df2.query('W < 20').groupby(by=['dn','n']).W.describe()

class y_axis_struct():
	def __init__(self):
		self.fields = {"x_values", "y_values", "y_err_1" "y_err_2",
		self.label = "default y_axis_struct",
		self.color = 'black'}
		return self


# def versus_n_plot(*y_axis_struct_list):
def versus_n_plot(df):
	'''Example Usage:
	df2g = df2.query('W < 20').groupby(by=['dn','n']).W.describe()
	'''
fontsize=20
figsize=(6,5)
list_W_n = [-2,2]
color_list = ['blue', 'red']

dn_list = list_W_n
assert(len(dn_list)==len(color_list))

# plot birth death rates with IQR y error bars with n on the x axis
fig, ax = plt.subplots(figsize=figsize
for dn, color in zip(dn_list,color_list):
    df2g = df.copy
    x_values, y_values, y_err_1, y_err_2 = df2g.loc[dn].dropna().reset_index()[['n', '50%','25%','75%']].values.T
    x_scatter_values, y_scatter_values = df2.query(f'dn=={dn}')[['n','W']].values.T
    yerr = np.array(list(zip(y_err_1,y_err_2))).T


    ax.scatter(x=x_scatter_values,y=y_scatter_values, c=color, s=10,alpha=0.5, label=f'$W_{{{int(dn)}}}$, domain size = {int(ds**2)} cm$^2$')
    ax.errorbar(x_values, y_values, yerr=yerr, c=color)

# ax.set_xticks(xticks)
# ax.set_yticks([0,10,20,30,40,50])
ymin, ymax = ax.get_ylim()
ax.set_ylim((1e-3,ymax+25))

ax.legend(loc='lower right', fontsize= fontsize-8)
ax.tick_params(axis='both', labelsize= fontsize)
ax.set_ylabel('birth/death rate (ms$^{-1}$)', fontsize=fontsize)
ax.set_xlabel('n', fontsize=fontsize) 
ax.set_yscale('log')
ax.set_title(f'$\sigma = {sigma}$, threshold = {threshold}', fontsize=fontsize)

# ax.set_title('high frequency birth deaths observed', fontsize=fontsize)
# ax.axis([20,60,0,7])
plt.tight_layout()
fig.savefig(f"{nb_dir}/Figures/birth_deaths_zero_odd_births-deaths_{descrip}.png",dpi=400)

# fig.savefig('Figures/birth_deaths_zero_odd_births-deaths_ds_5_pbc.pdf')
# fig.savefig(f'Figures/birth_deaths_zero_odd_births-deaths_ds_{ds}_pbc.png',dpi=400)