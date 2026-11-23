from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parent.parent/'03_robustness_and_coverage'/'lopzo_coverage'
OUT=Path(__file__).resolve().parent/'risk_coverage_results'
d=pd.read_csv(ROOT/'zone_summary.csv'); assert len(d)==36 and not d.duplicated(['target','zone_id']).any()
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
panels=[('A',r'$A$  Elongation','percentage points'),('Ak',r'$A_k$  Impact toughness',r'J cm$^{-2}$'),
        ('Rm',r'$R_m$  Tensile strength','MPa'),('Re',r'$R_e$  Yield strength','MPa')]
fig,axes=plt.subplots(4,2,figsize=(7.2,11.6),layout='constrained')
for i,(target,title,unit) in enumerate(panels):
    sub=d[d.target==target].set_index('zone_id');assert set(sub.index)==set(range(9))
    for j,(column,cmapname,legend) in enumerate([('percentile_median','Blues','Reference percentile'),('mae','YlOrRd',f'MAE ({unit})')]):
        ax=axes[i,j]; vals=np.array([sub.loc[z,column] for z in range(9)]).reshape(3,3)
        norm=Normalize(0,1 if j==0 else vals.max());cmap=mpl.colormaps[cmapname];im=ax.imshow(vals,origin='lower',norm=norm,cmap=cmap)
        for z in range(9):
            y,x=divmod(z,3);row=sub.loc[z];v=vals[y,x];color='white' if norm(v)>.57 else '#222222'
            sparse=row['n']<5 or row['n_compositions']<2
            ax.text(x,y+.17,f'{v:.2f}' if j==0 else f'{v:.1f}',ha='center',va='center',color=color,weight='bold',fontsize=9)
            ax.text(x,y-.14,f'Z{z}{"*" if sparse else ""}  n={int(row["n"])}',ha='center',va='center',color=color,fontsize=7)
        ax.set_xticks([0,1,2],['Q1','Q2','Q3']);ax.set_yticks([0,1,2],['Q1','Q2','Q3'])
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',bottom=False,left=False)
        ax.set_xlabel(r'$\tau$ (short $\rightarrow$ long)');ax.set_ylabel(r'$\Delta T_I$ (low $\rightarrow$ high)')
        ax.set_title('Training-reference distance' if j==0 else 'Held-out prediction error',loc='left',fontsize=9,fontweight='bold')
        cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03,shrink=.82);cb.set_label(legend,fontsize=7);cb.ax.tick_params(labelsize=7)
    axes[i,0].text(-.39,1.20,'abcd'[i]+'  '+title,transform=axes[i,0].transAxes,ha='left',va='bottom',fontweight='bold',fontsize=10)
fig.text(.5,-.01,'Cells: median reference distance percentile (left) or MAE (right); n: observations.',ha='center',fontsize=7.5)
fig.text(.5,-.023,'* Fewer than 5 observations or fewer than 2 compositions. Scales differ by metric and property.',ha='center',fontsize=7.5)
fig.savefig(OUT/'coverage_error_heatmap_comparison.svg',bbox_inches='tight',facecolor='white')
fig.savefig(OUT/'coverage_error_heatmap_comparison.pdf',bbox_inches='tight',facecolor='white')
fig.savefig(OUT/'coverage_error_heatmap_comparison.png',dpi=600,bbox_inches='tight',facecolor='white')
d.to_csv(OUT/'coverage_error_heatmap_source.csv',index=False,encoding='utf-8-sig')
print('Saved to',OUT)
