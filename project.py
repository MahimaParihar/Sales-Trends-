import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def generate_sales_data(years=3):  # changed to 3 years to include 2023
    np.random.seed(42)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    data = []
    current_year = datetime.now().year
    for year in range(current_year - years + 1, current_year + 1):
        for i, month in enumerate(months):
            electronics = 10000 + 5000 * np.sin(i/3) + np.random.normal(0, 1000)
            clothing = 8000 + 3000 * np.cos(i/2) + np.random.normal(0, 800)
            grocery = 15000 + 2000 * (i % 6) + np.random.normal(0, 500)

            record = {
                'Year': year,
                'Month': month,
                'MonthNum': i + 1,
                'Date': pd.to_datetime(f"{year}-{i + 1}-1"),
                'Electronics': max(0, electronics),
                'Clothing': max(0, clothing),
                'Grocery': max(0, grocery)
            }
            data.append(record)

    df = pd.DataFrame(data)
    df['Total Sales'] = df[['Electronics', 'Clothing', 'Grocery']].sum(axis=1)
    return df

# Generate data including 2023
sales_df = generate_sales_data(years=3)
print(sales_df.head())

def plot_sales_trend(df, highlight_year=None):
    plt.figure(figsize=(12, 6))

    monthly_avg = df.groupby('MonthNum')['Total Sales'].mean()

    if highlight_year:
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            plt.plot(year_data['MonthNum'], year_data['Total Sales'],
                     marker='o', linestyle='--', alpha=0.2,
                     label=f'{year}' if year != highlight_year else None)

        highlight_data = df[df['Year'] == highlight_year]

        if highlight_data.empty:
            print(f"Warning: Year {highlight_year} not found in data.")
            plt.text(0.5, 0.5, f"No data for {highlight_year}", ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.show()
            return

        plt.plot(highlight_data['MonthNum'], highlight_data['Total Sales'],
                 marker='o', linestyle='-', color='blue', label=f'{highlight_year}', linewidth=2)

        plt.plot(monthly_avg.index, monthly_avg.values,
                 marker='o', linestyle='-', color='green', label='Average Trend', alpha=0.8)

        max_row = highlight_data.loc[highlight_data['Total Sales'].idxmax()]
        min_row = highlight_data.loc[highlight_data['Total Sales'].idxmin()]
        max_sales = max_row['Total Sales']
        min_sales = min_row['Total Sales']
        avg_sales = highlight_data['Total Sales'].mean()
    else:
        plt.plot(monthly_avg.index, monthly_avg.values,
                 marker='o', linestyle='-', color='green',
                 label='Average Trend', linewidth=2)
        max_row = df.loc[df['Total Sales'].idxmax()]
        min_row = df.loc[df['Total Sales'].idxmin()]
        max_sales = max_row['Total Sales']
        min_sales = min_row['Total Sales']
        avg_sales = df['Total Sales'].mean()

    plt.title('Monthly Sales Trend\n(With Yearly Comparison)', pad=20)
    plt.xlabel('Month', labelpad=10)
    plt.ylabel('Sales (USD)', labelpad=10)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    plt.axhline(avg_sales, color='red', linestyle=':', alpha=0.5)
    plt.text(12.2, avg_sales, f'Avg: ${avg_sales:,.0f}',
             va='center', color='red')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    plt.annotate(f'{"Year " + str(highlight_year) if highlight_year else "Overall"} Analysis\n'
                 f'Peak: ${max_sales:,.0f} in {max_row["Month"]}\n'
                 f'Low: ${min_sales:,.0f} in {min_row["Month"]}',
                 xy=(0.5, -0.15), xycoords='axes fraction',
                 ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.2))
    plt.show()

plot_sales_trend(sales_df, highlight_year=2023)

def plot_category_comparison(df, selected_month=None, selected_year=None):
    if selected_month and selected_year:
        month_data = df[(df['Month'] == selected_month) & (df['Year'] == selected_year)]

        if month_data.empty:
            print(f"Warning: No data found for {selected_month} {selected_year}.")
            return

        title = f"Category Sales - {selected_month} {selected_year}"
        categories = ['Electronics', 'Clothing', 'Grocery']
        values = month_data[categories].values[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, y=1.05, fontsize=14)

        bars = ax1.bar(categories, values, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        ax1.set_title('Sales by Category')
        ax1.set_ylabel('Sales (USD)')

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'${height:,.0f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom')

        wedges, texts, autotexts = ax2.pie(values, labels=categories,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=['#1f77b4', '#ff7f0e', '#2ca02c'])

        ax2.set_title('Sales Distribution')
        ax2.axis('equal')

        ax2.legend(wedges, categories, title='Categories',
                   loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(14, 7))

        pivot_df = df.pivot(index='MonthNum', columns='Year',
                            values=['Electronics', 'Clothing', 'Grocery'])
        for i, category in enumerate(['Electronics', 'Clothing', 'Grocery']):
            plt.subplot(3, 1, i + 1)
            for year in df['Year'].unique():
                plt.plot(pivot_df[(category, year)].index,
                         pivot_df[(category, year)].values,
                         label=str(year))

            plt.title(f'{category} Sales Trend')
            plt.ylabel('Sales (USD)')
            plt.xticks(range(1, 13),
                       ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

            plt.legend()
            plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

# Call both functions correctly
plot_category_comparison(sales_df, selected_month='Jun', selected_year=2023)
plot_category_comparison(sales_df)



def plot_regional_analysis(df):
    fig=plt.figure(figsize=(14,8))

    gs=fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])

    plt.subplots_adjust(bottom=0.25)
    
    latest_date=df['Date'].max()
    current_data = df[df['Date']==latest_date]
    regions = ['North','South','East','West']
    region_colors = plt.cm.Pastel1(range(len(regions)))

    pie_wedges,pie_texts, pie_autotexts = ax1.pie(
        current_data[regions].values[0],
        labels=regions,
        autopct='%1.1f%%',
        startangle=90,
        colors=region_colors
    )

    ax1.set_title(f'Regional Distribution - {current_data['Month'].values[0]}{current_data['Year'].values[0]}')

    bar_rects = ax2.bar(regions, current_data[regions].values[0],
                        color=region_colors)
    ax2.set_title('Regional Sales Comparison')
    ax2.set_ylabel('Sales(USD)')

    for rect in bar_rects:
        height = rect.get_height()

        ax2.annotate(f'${height:,.0f}',
                     xy=(rect.get_x()+rect.get_width()/2,height),
                     xytext='center',va='bottom')
    
    for i, region in enumerate(regions):
        ax3.plot(df['Date'],df[region],
                 label=region,color=region_colors[i])
        ax3.set_title('Regional Sales Trend Over Time')
        ax3.set_ylabel('Sales(USD)')
        ax3.legend()
        ax3.grid(alpha=0.3)

        ax_radio = plt.axes([0.25,0.1,0.5,0.1])
        month_selector = RadioButtons(ax_radio,labels=sorted(df['Month'].unique()),active=df['MonthNum'].max()-1)

        def update_plots(month):
            selected_data = df[(df['Month']==month) & (df['Year']==df['Year'].max())]

            ax1.clear()
            pie_wedges,pie_texts,pie_autotexts=ax1.pie(selected_data[regions].values[0],
                                                       labels=regions,
                                                       autopct='%1.1f%%',
                                                       startangle=90,
                                                       colors=region_colors)
            
            ax1.set_title(f'Regional Distribution - {month}{selected_data['Year'].value[0]}')
            ax2.clear()
            bar_rects = ax2.bar(regions,selected_data[regions].values[0],
                                color=region_colors)
            ax2.set_title('Regional Sales Comparison')
            ax2.set_ylabel('Sales(USD)')

            for rect in bar_rects:
                height = rect.get_height()
                ax2.annotate(f'${height:,.0f}',
                             xy=(rect.get_x()+rect.get_width()/2,height),
                             xytext=(0,3),textcoords='offset points',
                             ha='center',va='bottom')
                
                fig.canvas.draw_idle()
                month_selector.on_clicked(update_plots)

                plt.tight_layout()
                plt.show()

            plot_regional_analysis(sales_df)
            def create_sales_dashboard(df):
                fig = plt.figure(figsize=(18,12))
                fig.suptitle('COMPANY SALES DASHBOARD',y=0.98,fontsize=16,fontweight='bold')

                gs=fig.add_gridspec(3,3)

                ax1 = fig.add_subplot(gs[0,:2])
                ax2 = fig.add_subplot(gs[1,:2])
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
                ax5 = fig.add_subplot(gs[:,2])

                plt.subplots_adjust(bottom=0.15, hspace=0.5, wspace=0.4)

                years = df['Year'].unique()
                for year in years:
                    year_data = df[df['Year']==year]
                    ax1.plot(year_data['MonthNum'],year_data['Total Sales'],
                             marker='o',linestyle='--' if year != years[-1] else'-',
                             alpha=0.7 if year != years[-1] else 1,
                             linewidth = 1 if year!=years[-1] else 2,
                             label = str(year))
                    ax1.set_title('Monthly Sales Trend', pad=10)
                    ax1.set_xlabel('Month',labelpad=5)
                    ax1.set_ylabel('Sales(USD)', labelpad=5)
                    ax1.set_xticks(range(1,13))
                    ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(title='Year')

                    current_year = year[-1]
                    prev_year = years[-2] if len(years)>1 else None

                    ytd_current = df[df['Year']==current_year]['Total Sales'].sum()
                    if prev_year:
                        ytd_prev = df[df['Year']==prev_year]['Total Sales'].sum()
                        growth = (ytd_current - ytd_prev)/ytd_prev*100
                        ax1.annotate(f'YTD Growth:{growth:+.1f}% vs {prev_year}',
                                     xy=(0.5,-0.15),xycoords='axes fraction',
                                     ha = 'center',va='center', bbox = dict(boxstyle='round',alpha=0.2))

                    pivot_cat = df.pivot(index='MonthNum',columns = 'Year', values=['Electronics','Clothing','Grocery'])

                    for i, category in enumerate(['Electronics','Clothing','Grocery']):
                        ax2.fill_between(pivot_cat.index,
                                         pivot_cat[(category,current_year)].values,
                                         label=category,
                                         alpha=0.7)
                        ax2.set_title('Category Sales Composition', pad=10)
                        ax2.set_xlabel('Month',labelpad=5)
                        ax2.set_ylabel('Sales(USD)',labelpad=5)
                        ax2.set_xticks(range(1,13))
                        ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
                        ax2.grid(True,alpha=0.3)
                        ax2.legend() 


                        latest_date = df[df['Date']==df['Date'].max()]
                        regions = ['North','South','East','West']
                        region_colors = plt.cm.Pastel1(range(len(region)))

                        wedges,texts,autotexts = ax3.pie(
                            latest_data[regions].values[0],
                            labels = regions,
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=region_colors
                        ) 

                        ax3.set_title(f'Regional Distribution\n{latest_data['Month'].values[0]}{latest_data['Year'].values[0]}',pad=10)

                        ytd_regions = df[df['Year']==current_year][regions].sum()
                        if prev_year:
                            ytd_regions_prev = df[df['Year']==prev_year][regions].sum()

                        bars = ax4.bar(regions,ytd_regions,color=region_colors)
                        ax4.set_title('Regional Sales YTD Comparison', pad=10)
                        ax4.set_ylabel('Sales(USD)')

                        for bar in bars:
                            height = bar.get_height()
                            ax4.annotate(f'${height:,.0f}',
                                         xy=(bar.get_x()+bar.get_width()/2,height),
                                         xytext=(0,3),textcoords='offset points',
                                         ha='center',va='bottom')
                            
                        ax5.axis('off')

                        total_sales = df['Total Sales'].sum()
                        avg_monthly = df['Total Sales'].mean()
                        best_month = df.loc[df['Total Sales'].idxmax()]
                        worst_month = df.loc[df['Total Sales'].idxmin()]

                        kpi_text = (
                            f'SALES KPIs SUMMARY\n\n'
                            f'Total Sales(All Time):\n${total_sales:,.0f}\n\n'
                            f'Best Performing Month:\n{best_month['Month']}{best_month['Year']}\n'
                            f'${best_month['Total Sales']:,.0f}\n\n'
                            f'Worst Performing Month:\n{worst_month['Month']}{worst_month['Year']}\n'
                            f'${worst_month['Total Sales']:,.0f}'
                        )

                        ax5.text(0.5,0.5,kpi_text,ha='center',va='center',fontsize=12,bbox=dict(facecolor='whitesmoke',alpha=0.5))

                        ax_radio = plt.axes([0.3,0.05,0.4,0.05])
                        year_selector = RadioButtons(
                            ax_radio,
                            labels = [str(year) for year in years],
                            active = len(years)-1
                        )

                        def update_dashboard(year):
                            year = int(year)

                            for line in ax1.lines:
                                line.set_alpha(0.3 if int(line.get_label()) !=year else 1)
                                line.set_linewidth(1 if int(line.get_label()) !=year else 2)

                            ax2.clear()
                            for i,category in enumerate(['Electronics','Clothing','Grocery']):
                                ax2.fill_between(pivot_cat.index,
                                                 pivot_cat[(category,year)].values,
                                                 label = category,
                                                 alpha = 0.7)
                                
                                ax2.set_title('Category Sales Composition', pad=10)
                                ax2.set_xlabel('Month', labelpad=5)
                                ax2.set_ylabel('Sales(USD)',labelpad=5)
                                ax2.set_xticks(range(1,13))
                                ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
                                ax2.grid(True,alpha = 0.3)
                                ax2.legend()

                                ax4.clear()
                                ytd_regions = df[df['Year']==year][regions].sum()
                                bars = ax4.bar(regions, ytd_regions, color=region_colors)

                                ax4.set_title(f'Regional Sales YTD Comparison -{year}', pad=10)
                                ax4.set_ylabel('Sales(USD)')

                                for bar in bars:
                                    height = bar.get_height()
                                    ax4.annotate(f'${height:,.0f}',
                                                 xy=(bar.get_x()+bar.get_width()/2,height),
                                                 xytext=(0,3), textcoords='offset points',
                                                 ha = 'center', va='bottom')
                                    
                                    fig.canvas.draw_idle()
                                year_selector.on_clicked(update_dashboard)

                                plt.tight_layout()
                                plt.show()
            create_sales_dashboard(sales_df)

            def export_dashboard(df,filename = 'sales_dashboard.png', dpi=300):
                """Export the dashboard as a high-resolution image"""
                fig = plt.figure(figsize=(18,12), dpi=dpi)

                plt.savefig(filename, bbox_inches='tight',dpi=dpi)
                plt.close()
                print(f'Dashboard exported to {filename}')

                export_dashboard(sales_df,'my_sales_dashbord.png')

                             



                    