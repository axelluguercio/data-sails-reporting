#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import re 
from matplotlib import pyplot as plt
import numpy as np
import math
import fdb
from dotenv import load_dotenv
import os
from fpdf import FPDF
import datetime

# In[40]:


# SQL connection (!!!Hay que estar conectado a la VPN!!!)
con = fdb.connect(
    dsn=os.environ["DB"],
    user=os.environ["USERNAME"], password=os.environ["PASSWORD"],
    charset='UTF-8'
  )

# SQL sentence for 118
sql_118 = '''SELECT Q$VTAS_ART_UTIL.*,
       CLIENTES.COD_EXTERNO,
       CLIENTES.NOM_FANTASIA,
       CPR.IVA_ID
FROM Q$VTAS_ART_UTIL(?, ?, ?, ?, ?, ?)
JOIN CPR ON (CPR.CPR_ID = Q$VTAS_ART_UTIL.CPR_ID)
JOIN CLIENTES ON (CLIENTES.CLI_ID = CPR.NRO_ENT_ID)'''

# SQL sentence for 580
sql_580 = '''SELECT Q$VTAS_NETAS_X_LPR.*
FROM Q$VTAS_NETAS_X_LPR(?, ?, ?)'''

# In[41]:


# Parametros de entrada
emp_id = 3
usuario_id = 56
# Tomar las fechas desde variables de entorno
load_dotenv()
fecha_inicio = os.environ["INICIO"]
fecha_fin = os.environ["FIN"]

print(fecha_fin)

# In[42]:


# crear el objeto cursor:
cur = con.cursor()

# Execute 118
cur.execute(sql_118, (emp_id, usuario_id, fecha_inicio, fecha_fin, 'N', 'S'))

# Get columns
column_names = [i[0] for i in cur.description]

# Convert results to a DataFrame
df_result = pd.DataFrame(cur.fetchall(), columns=column_names)

# Filtro 118
columns_todrop = ['CPR_ID', 'CPR_CLASIF_ID', 'MARCA_ID', 'MARCA', 'LETRA_CPR', 'CP', 'LOCALIDAD', 'PROVINCIA', 'TIPO_PAGO_ID', 'COND_ID', 'PORC_FINAN', 'DISCRIMINA_PORC', 'CPRDET_ID', 'ART_ID', 'MOD', 'MED', 'ART_CLASIF', 'PR_COSTO_CPRA', 'PR_COSTO', 'PR_VTA', 'PR_NETO', 'PR_NETO_UNIT', 'ESCALA', 'EQUIVALENCIA', 'CANT_EQUIVALENCIA', 'ART_EQUIVALENCIA', 'COSTO', 'COSTO_CPRA', 'COSTO_CPRA_TOTAL', 'COSTO_TOTAL', 'UTIL_BRUTA', 'UTIL_BRUTA_CPRA', 'UTIL_TOTAL_CPRA', 'UTILIDAD', 'UTILIDAD_CPRA', 'LEGAJO', 'ZONA', 'CIRCUITO', 'ART_GRUPO_ID', 'EMPL_ID', 'IVA_ID']

columns_todrop_580 = ['LPR_CLASIF', 'PORC']

# 118 filtrado result
df = df_result.drop(columns_todrop, axis=1)

# Execute 580
cur.execute(sql_580, (emp_id, fecha_inicio, fecha_fin))

# Get columns
column_names = [i[0] for i in cur.description]

# Convert results to a DataFrame
df_listas = pd.DataFrame(cur.fetchall(), columns=column_names)

df_listas = df_listas.drop(columns_todrop_580, axis=1)

# Close connection
con.commit()

# In[ ]:


familias = ['ALIMENTOS', 'GOLOSINAS', 'LIMPIEZA PERFUMERIA', 'VARIOS', 'REFRIGERADOS']

vendedores_olav = ['DARIO DELIBANO', 'DIAZ GUSTAVO JAVIER', 'SANCHEZ JAVIER', 'STEIMBACH NICOLAS']

vendedores_bol = ['BUGLIONI MARIO', 'ELORDIETA DANIEL']

vendedores_tot = ['BUGLIONI MARIO', 'DARIO DELIBANO', 'DIAZ GUSTAVO JAVIER', 'ELORDIETA DANIEL',
                  'SANCHEZ JAVIER', 'STEIMBACH NICOLAS']

index = ['TOTAL']

# planes mensuales

planes = {'BUGLIONI MARIO': 3200000, 'DARIO DELIBANO': 3600000, 'DIAZ GUSTAVO JAVIER': 4500000, 'ELORDIETA DANIEL': 3600000,
           'SANCHEZ JAVIER': 2800000, 'STEIMBACH NICOLAS': 4900000}

# clientes mensuales

clientes = {'BUGLIONI MARIO': 110, 'DARIO DELIBANO' : 144, 'DIAZ GUSTAVO JAVIER': 109, 'ELORDIETA DANIEL': 100,
             'SANCHEZ JAVIER': 106, 'STEIMBACH NICOLAS': 182}

# DIAS HABILES

dias_habiles = 26

#

montos = [7000, 7000, 7000, 7000, 7000, 7000]
items = [11, 11, 11, 11, 11, 11]

# In[ ]:


# Imprime los dias en pasados en la tabla

print(df['FEC_EMISION'].nunique())

# agrupa por punto de venta

# Olavarria

df_olavarria = df[df['NRO'].isin([10, 27])]

# Bolivar

df_bolivar = df[df['NRO'].isin([11, 28])]

# In[ ]:


def sacarTablaVendedor(tabla, vendedor):
    
    df_pivot = tabla.loc[tabla['VENDEDOR'].str.contains(vendedor, na=False)]
    
    return df_pivot

def obtenerTablas(tabla, vendedores):

    serie_tablas = []

    for vendedor in vendedores :
    
        serie_tablas.append(sacarTablaVendedor(tabla, vendedor))
        
    df_result = pd.concat(serie_tablas, ignore_index=True)
    
    return df_result

def concatenarTablas(tabla1, tabla2):
    
    df_concat = pd.concat([tabla1, tabla2], ignore_index=True)
    
    return df_concat

def juntarVendedores(list_vendedores1, list_vendedores2):
    
    list_final = []
    
    for i in range(0, len(list_vendedores1)):
        
        list_final.append(list_vendedores1[i])
    
    for i in range(0, len(list_vendedores2)):
        
         list_final.append(list_vendedores2[i])
        
    return list_final

def verificarVendedores(tabla, vendedores):
    
    vendedores_presentes = []
    
    grupo = tabla.groupby('VENDEDOR')
    
    for vendedor in vendedores:
        
        try:
            df_vendedor = grupo.get_group(vendedor)
             
            vendedores_presentes.append(vendedor)
            
        except KeyError:
            
            print(vendedor, 'no facturo')
            
    return vendedores_presentes

# In[ ]:


df_olavarria_filt = obtenerTablas(df_olavarria, vendedores_olav)
df_bolivar_filt = obtenerTablas(df_bolivar, vendedores_bol)

df_final = concatenarTablas(df_olavarria_filt, df_bolivar_filt)

df_ventas = pd.DataFrame(df_final.groupby('VENDEDOR')['VTA_TOTAL'].sum()).astype(int)

# In[ ]:


try:
    df_ventas.plot.pie(subplots=True,
                       shadow= False,
                       autopct='%1.1f%%',
                       figsize=(10,9))

    plt.title('REPRESENTACION EN VENTAS')
    plt.savefig('torta_venta', bbox_inches='tight', dpi=200)
    
except ValueError:
    print("valores no representativos para pie charts skipping....")

dias_transcurridos = df_final['FEC_EMISION'].nunique()

# In[ ]:


## junta vendedores

vendedores_reales = verificarVendedores(df_final, vendedores_tot)

vendedores_reales.sort()

objetivos_currentes = []

for vendedor in vendedores_reales:
     
    objetivos_currentes.append((planes[vendedor] / dias_habiles) * dias_transcurridos)  
    
df_ventas['PROYECCIÓN'] = objetivos_currentes

df_ventas.plot.bar(rot=0,
               figsize=(15,6))

plt.grid(color='#e62e1b', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.title('VENTA TOTAL VENDEDOR COMPARADA A LA PROYECCIÓN')
plt.savefig('venta_tot', bbox_inches='tight', dpi=200)

# In[ ]:




# In[ ]:


# Estadisticas

def sacarEstadisticas(tabla, vendedores):
    
    grupos = []
    grupox = tabla.groupby('VENDEDOR')
    
    for familia in familias:
      
        grupoxvendedor = []
                                                      
        for vendedor in vendedores:
            
            df_vendedor = grupox.get_group(vendedor)
            
            df_familiafilt = df_vendedor.loc[df_vendedor['FAMILIA'].str.contains(familia, flags=re.I, na=False)]
            grupoxvendedor.append(int(df_familiafilt['VTA_TOTAL'].sum()))
            
        grupos.append(grupoxvendedor)
        
    return grupos

# In[ ]:


serie_grupo = sacarEstadisticas(df_final, vendedores_reales)

data = {'ALIMENTOS': serie_grupo[0], 'GOLOSINAS': serie_grupo[1], 'LIMP Y PERF': serie_grupo[2], 'VARIOS': serie_grupo[3], 'REFRIGERADOS': serie_grupo[4]}
df_ventas_grupo = pd.DataFrame(data, index=vendedores_reales, columns=['ALIMENTOS', 'GOLOSINAS', 'LIMP Y PERF', 'VARIOS', 'REFRIGERADOS'])

df_ventas_grupo.plot.bar(rot=0,
                    figsize=(15,10))

plt.grid(color='#e62e1b', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.title('VENTAS POR FAMILIA POR VENDEDOR')
plt.savefig('data/detalle_venta_por_grupo', bbox_inches='tight', dpi=150)

# In[ ]:


##### plan venta

serie_plan = []

for clave, valor in planes.items():
    
    serie_plan.append(valor)

df_ventas['PLAN'] = serie_plan

porcentaje = []

for i in range(0, len(serie_plan)):
    
    try:
        
        porcentaje.append(math.trunc(df_ventas['VTA_TOTAL'][i] / serie_plan[i]*100))
    
    except(OverflowError):
        
        porcentaje.append(0)

df_ventas['% PLAN'] = porcentaje

# In[ ]:


## render para las tablas 
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, rowLabels=data.index, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    
    w, h = mpl_table[0,1].get_width(), mpl_table[0,1].get_height()
    mpl_table.add_cell(0, -1, w,h, text=data.index.name)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

# In[ ]:


# clientes vendedores
def sacar_clientes(df, vendedor):
    clientes = []
    df_vendedor = df.loc[df['VENDEDOR'] == vendedor]
    # ciclo
    for index, row in df_vendedor.iterrows():
        if row['TIPO_CLI'] == 'LISTA ESPECIAL CHINOS' or row['TIPO_CLI'] == 'CHINO BASE':
                if row['RZ'] == 'CONS. FINAL': 
                    clientes.append(row['NOM_FANTASIA'])
                else:
                    clientes.append(row['RZ'])
        else:
            if row['RZ'] == 'CONS. FINAL':
                clientes.append(row['COD_EXTERNO'])
            else:
                clientes.append(row['COD_CLI'])
    return clientes

# In[ ]:


# por cada vendedor
tablas_clientes = []
for vendedor in vendedores_reales:
    vend = []
    codigos = sacar_clientes(df_final, vendedor)
    for i in range(0, len(codigos)):
        vend.append(vendedor)
    df_vendedor = pd.DataFrame({'COD_CLI': codigos, 'VENDEDOR': vend})
    tablas_clientes.append(df_vendedor)

df_clientes_concat = pd.concat(tablas_clientes, ignore_index=True)

df_clientes = pd.DataFrame(df_clientes_concat.groupby('VENDEDOR').nunique()['COD_CLI'])

# In[ ]:


clientes_serie = []

for clave, valor in clientes.items():
        
    clientes_serie.append(valor)

vendedores_tot.sort()

df_clientes['RUTA'] = clientes_serie

df_clientes.rename(columns={'COD_CLI': 'CLIENTES VENDIDOS'}, inplace=True)

porcentaje = []

for i in range(0, len(clientes_serie)):
    
    porcentaje.append(math.trunc(df_clientes['CLIENTES VENDIDOS'][i] / clientes_serie[i]*100))
    
df_clientes['CUMPLIMIENTO'] = porcentaje

df_clientes_graf = df_clientes.filter(['CLIENTES VENDIDOS', 'RUTA'])

df_clientes_graf.plot.bar(rot=0,
                    figsize=(15,10))
plt.grid(color='#e62e1b', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.title('CLIENTES VENDIDOS FRENTE AL PLAN')
plt.savefig('clientes_grafico', bbox_inches='tight', dpi=250)

# In[ ]:


# promedios facturas e items

def sacarEstadisticas(tabla, vendedores, lista, promxitem, promxfact, listaClientes, listaFacturas):
    
    items = []
    ventas = []
    
    grupox = tabla.groupby('VENDEDOR')
    
    for vendedor in vendedores:
      
        df_vendedor = grupox.get_group(vendedor)
        
        df_vendedor_clientes = df_vendedor.loc[df_vendedor['CPR_TIPO_ID'] == 'FA']
        
        # Calculo los clientes visitados y facturados
        
        listaClientes.append(df_vendedor_clientes['RZ'].nunique())
        listaFacturas.append(df_vendedor_clientes['NRO_CPR'].nunique())
    
        item = []
        unidad = []
        venta = []
        utilidad = []
                                                      
        for familia in familias:
        
            df_familiafilt = df_vendedor.loc[df_vendedor['FAMILIA'].str.contains(familia, flags=re.I, na=False)]

            item.append(df_familiafilt['DESCRIPCION'].nunique())
            unidad.append(int(df_familiafilt['CANT'].sum()))
            venta.append(int(df_familiafilt['VTA_TOTAL'].sum()))
            utilidad.append(int(df_familiafilt['UTIL_TOTAL'].sum()))
        
        data = {'ITEM': item, 'UNIDAD': unidad, 'VENTA': venta, 'UTILIDAD': utilidad}
        df_pivot = pd.DataFrame(data, index=familias, columns=['ITEM', 'UNIDAD', 'VENTA', 'UTILIDAD'])
        
        lista.append(df_pivot)
    
    for i in range(0, len(lista)):
    
        item = lista[i]['ITEM'].sum()
        venta = lista[i]['VENTA'].sum()
    
    
        items.append(item)
        ventas.append(venta)
    
        #calculo el promedio
        
        promxitem.append(items[i] / listaClientes[i])
        promxfact.append(ventas[i] / listaFacturas[i])

tablas_info = []

listaProm = []
listaItem = []

lista_cli_visit = []
lista_cli_fact = []

sacarEstadisticas(df_final, vendedores_reales, tablas_info, listaItem, listaProm, lista_cli_visit, lista_cli_fact)

data_0 = {'PROM ITEMS': listaItem, 'PROM FACT': listaProm}
df_prom = pd.DataFrame(data_0, index=vendedores_reales, columns=['PROM ITEMS', 'PROM FACT'])

df_prom['MONTO OBJ'] = montos
df_prom['ITEMS OBJ'] = items

ay = df_prom['ITEMS OBJ'].plot.line(color='r')

df_prom['PROM ITEMS'].plot.bar(rot=0,
           figsize=(15,6), ax=ay)

plt.title('PROMEDIO ITEMS POR FACTURA')
plt.savefig('data/prom_items', bbox_inches='tight', dpi=150)

# objetivos a tabla

ax = df_prom['MONTO OBJ'].plot.line(color='r')

df_prom['PROM FACT'].plot.bar(rot=0,
            figsize=(15,6), ax=ax)

plt.title('PROMEDIO MONTO POR FACTURA')
plt.savefig('data/prom_fact', bbox_inches='tight', dpi=150)

# In[ ]:


# totales

# clientes

df_clientes_tot = pd.DataFrame(index=index)

cli_tot = df_clientes['CLIENTES VENDIDOS'].sum()
ruta_tot = df_clientes['RUTA'].sum()
porc = df_clientes['CUMPLIMIENTO'].mean()

df_clientes_tot['CLIENTES VENDIDOS'] = cli_tot
df_clientes_tot['RUTA'] = ruta_tot
df_clientes_tot['CUMPLIMIENTO'] = porc

df_venta_clientes_tot = pd.concat([df_clientes, df_clientes_tot], axis=0, ignore_index=False)

# In[ ]:


# ventas

df_venta_tot = pd.DataFrame(index=index)

venta_tot = df_ventas['VTA_TOTAL'].sum()
venta_pro = df_ventas['PROYECCIÓN'].sum()
venta_plan = df_ventas['PLAN'].sum()
porc = round((venta_tot/venta_plan)*100, 2)

df_venta_tot['VTA_TOTAL'] = venta_tot
df_venta_tot['PROYECCIÓN'] = venta_pro
df_venta_tot['PLAN'] = venta_plan
df_venta_tot['% PLAN'] = porc

df_venta_vendedor_tot = pd.concat([df_ventas, df_venta_tot], axis=0, ignore_index=False)

# In[ ]:


# familias

df_venta_grupo_tot = pd.DataFrame(index=index)

venta_al = df_ventas_grupo['ALIMENTOS'].sum()
venta_gol = df_ventas_grupo['GOLOSINAS'].sum()
venta_lim = df_ventas_grupo['LIMP Y PERF'].sum()
venta_var = df_ventas_grupo['VARIOS'].sum()
venta_ref = df_ventas_grupo['REFRIGERADOS'].sum()

df_venta_grupo_tot['ALIMENTOS'] = venta_al
df_venta_grupo_tot['GOLOSINAS'] = venta_gol
df_venta_grupo_tot['LIMP Y PERF'] = venta_lim
df_venta_grupo_tot['VARIOS'] = venta_var
df_venta_grupo_tot['REFRIGERADOS'] = venta_ref

df_venta_familia_tot = pd.concat([df_ventas_grupo, df_venta_grupo_tot], axis=0, ignore_index=False)

# In[ ]:


df_productos_facturacion = pd.DataFrame(df_final.groupby('DESCRIPCION')['VTA_TOTAL'].sum()).astype(int)

df_productos_facturacion = df_productos_facturacion.sort_values(['VTA_TOTAL', 'DESCRIPCION'], ascending=False)

df_productos_facturacion = df_productos_facturacion.head(20)

###############################

df_productos_unidad = pd.DataFrame(df_final.groupby('DESCRIPCION')['CANT'].sum()).astype(int)

df_productos_unidad = df_productos_unidad.sort_values(['CANT', 'DESCRIPCION'], ascending=False)

df_productos_unidad = df_productos_unidad.head(20)

# In[ ]:


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, rowLabels=data.index, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

# In[ ]:


# cambio orden a la tabla venta

df_venta_vendedor_tot = df_venta_vendedor_tot.reindex(['PLAN', 'PROYECCIÓN', 'VTA_TOTAL', '% PLAN'], axis="columns")

df_venta_vendedor_tot['PLAN'] = df_venta_vendedor_tot['PLAN'].map('${:,.2f}'.format)
df_venta_vendedor_tot['PROYECCIÓN'] = df_venta_vendedor_tot['PROYECCIÓN'].map('${:,.2f}'.format)
df_venta_vendedor_tot['VTA_TOTAL'] = df_venta_vendedor_tot['VTA_TOTAL'].map('${:,.2f}'.format)
df_venta_vendedor_tot['% PLAN'] = df_venta_vendedor_tot['% PLAN'].map('{:.2f}%'.format)

fig,ax = render_mpl_table(df_venta_vendedor_tot, header_columns=0, col_width=3.0)
fig.savefig("data/tabla_ventasxvendedor.png", bbox_inches = 'tight', dpi=150)

# In[ ]:


# ventas por grupo

df_venta_familia_tot['ALIMENTOS'] = df_venta_familia_tot['ALIMENTOS'].map('${:,.2f}'.format)
df_venta_familia_tot['GOLOSINAS'] = df_venta_familia_tot['GOLOSINAS'].map('${:,.2f}'.format)
df_venta_familia_tot['LIMP Y PERF'] = df_venta_familia_tot['LIMP Y PERF'].map('${:,.2f}'.format)
df_venta_familia_tot['VARIOS'] = df_venta_familia_tot['VARIOS'].map('${:,.2f}'.format)
df_venta_familia_tot['REFRIGERADOS'] = df_venta_familia_tot['REFRIGERADOS'].map('${:,.2f}'.format)

fig,ax = render_mpl_table(df_venta_familia_tot, header_columns=0, col_width=3.0)
fig.savefig("data/tabla_ventasxgrupo.png", bbox_inches = 'tight', dpi=150)


# In[ ]:


# clientes

df_venta_clientes_tot['CUMPLIMIENTO'] = df_venta_clientes_tot['CUMPLIMIENTO'].map('{:.2f}%'.format)

# Reemplaza parada por buglioni
df_venta_clientes_tot = df_venta_clientes_tot.rename(index={'PARADA MATIAS':'BUGLIONI'})

fig,ax = render_mpl_table(df_venta_clientes_tot, header_columns=0, col_width=3.0)
fig.savefig("data/tabla_clientes.png",  bbox_inches = 'tight', dpi=150)

# In[ ]:


# productos mas vendidos por facturacion

df_productos_facturacion['VTA_TOTAL'] = df_productos_facturacion['VTA_TOTAL'].map('${:,.2f}'.format)

fig,ax = render_mpl_table(df_productos_facturacion, header_columns=0, col_width=4.0)
fig.savefig("data/tabla_productos_mas_vendidos.png",  bbox_inches = 'tight', dpi=150)

# In[ ]:


# productos mas vendidos por unidad

fig,ax = render_mpl_table(df_productos_unidad, header_columns=0, col_width=4.0)
fig.savefig("data/tabla_productos_mas_vendidos_porunidad.png",  bbox_inches = 'tight', dpi=150)

# In[ ]:


# lista precios

df_ofertas = df_listas[df_listas['LISTA_ID'].isin([10, 22])]

df_listas = df_listas[df_listas['LISTA_ID'].isin([7, 17])]

sum_ofertas = int(df_ofertas['VTA_NETA'].sum())

oferta = {'LISTA': 'OFERTAS', 'VTA_NETA': sum_ofertas}

df_listas = df_listas.append(oferta, ignore_index=True)

df_listas = df_listas.drop(['LISTA_ID'], axis=1)

# Suma las ofertas 

sum = int(df_listas['VTA_NETA'].sum())

porc = []

for venta in df_listas['VTA_NETA']:
    
    porc.append((int(venta)/sum)*100)

df_listas['% VENTA'] = porc

listas = [0, 1, 2]

labels = df_listas['LISTA']

df_listas['% VENTA'].plot.bar(listas, rot=0,
            figsize=(13,4))

plt.xticks(listas, labels)
plt.title('PORCENTAJE VENTA POR LISTA')
plt.grid(color='#e62e1b', linestyle='--', linewidth=1, axis='y', alpha=0.7)
plt.savefig('data/porc_listas', bbox_inches='tight', dpi=150)

# In[ ]:


today = datetime.date.today()
        
class PDF(FPDF):
    def header(self):
        # Logo
        self.image('sweet.png', 10, 8, 33)
        self.image('celu.jpg', 160, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'IO AUTOSERVICIOS Y EL CHOIQUE {}'.format(today), 0, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Pagina ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

print("CONVIRTIENDO LOS REPORTES EN PDF.....................%")

# Instantiation of inherited class
pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()
pdf.set_font('Arial', 'B', 12)

tabla_clientes = r"data/tabla_clientes.png"
clientes_grafico = r"data/clientes_grafico.png"

pdf.cell(10, 40, 'RUTA CLIENTES')

pdf.image(tabla_clientes, x = 10, y = 60, w = 180, h = 70, type = '')

pdf.image(clientes_grafico, x = 10, y = 180, w = 180, h = 100, type = '')

pdf.add_page()

pdf.cell(10, 40, 'VENTAS POR VENDEDOR')

tabla_ventas = r"data/tabla_ventasxvendedor.png"

pdf.image(tabla_ventas, x = 10, y = 60, w = 180, h = 70, type = '')

grafico_ventas = r"data/venta_tot.png"
grafico_promedio = r"data/prom_fact.png"
grafico_item = r"data/prom_items.png"

pdf.image(grafico_ventas, x = 10, y = 190, w = 180, h =90, type = '')


pdf.add_page()

pdf.image(grafico_promedio, x = 10, y = 40, w = 180, h =90, type = '')
pdf.image(grafico_item, x = 10, y = 160, w = 180, h = 90, type = '')

pdf.add_page

tablas_ventas_grupo = r"data/tabla_ventasxgrupo.png"

pdf.add_page()

pdf.cell(10, 40, 'VENTAS POR GRUPO POR VENDEDOR')

pdf.image(tablas_ventas_grupo, x = 10, y = 60, w = 180, h = 70, type = '')

grafico_venta_grupo = r"data/detalle_venta_por_grupo.png"

pdf.image(grafico_venta_grupo, x = 10, y = 140, w = 180, h = 110, type = '')

pdf.add_page()

grafico_torta = r"data/torta_venta.png"
grafico_listas = r"data/porc_listas.png"

pdf.cell(10, 20, 'CUMPLIMIENTO DEL PLAN')

pdf.image(grafico_torta, x = 10, y = 50, w = 180, h =130, type = '')
pdf.image(grafico_listas, x = 50, y = 180, w = 130, h =80, type = '')

pdf.add_page()

tabla_prod_unidad = r"data/tabla_productos_mas_vendidos_porunidad.png"

pdf.cell(10, 20, 'PRODUCTOS MAS VENDIDOS POR UNIDAD')

pdf.image(tabla_prod_unidad, x = 50, y = 50, w = 110, h =130, type = '')

pdf.add_page()

tabla_prod = r"data/tabla_productos_mas_vendidos.png"

pdf.cell(10, 20, 'PRODUCTOS MAS VENDIDOS POR FACTURACIÓN')

pdf.image(tabla_prod, x = 50, y = 50, w = 110, h =130, type = '')

print("LISTO...............##")

pdf.output('REPORTE_VENTAS {}.pdf'.format(today), 'F')
