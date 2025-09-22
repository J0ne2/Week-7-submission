import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("DATA ANALYSIS AND VISUALIZATION PROJECT")
print("="*60)

# Task 1: Load and Explore the Dataset
def load_and_explore_dataset():
    """Task 1: Load, explore, and clean the dataset"""
    print("\n" + "="*50)
    print("TASK 1: LOADING AND EXPLORING THE DATASET")
    print("="*50)
    
    try:
        # Load Iris dataset from sklearn
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        print("✅ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        
        # Display first few rows
        print("\nFirst 10 rows of the dataset:")
        print(df.head(10))
        
        # Explore dataset structure
        print("\n" + "-"*40)
        print("DATASET INFORMATION:")
        print("-"*40)
        print(df.info())
        
        print("\n" + "-"*40)
        print("DATA TYPES:")
        print("-"*40)
        print(df.dtypes)
        
        # Check for missing values
        print("\n" + "-"*40)
        print("MISSING VALUES:")
        print("-"*40)
        missing_values = df.isnull().sum()
        print(missing_values)
        
        # Clean the dataset (though Iris dataset is already clean)
        if missing_values.sum() > 0:
            print("\nCleaning dataset...")
            # Fill numerical missing values with mean
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
            
            # Fill categorical missing values with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            
            print("✅ Dataset cleaned successfully!")
        else:
            print("✅ No missing values found - dataset is clean!")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

# Task 2: Basic Data Analysis
def perform_basic_analysis(df):
    """Task 2: Perform basic statistical analysis"""
    print("\n" + "="*50)
    print("TASK 2: BASIC DATA ANALYSIS")
    print("="*50)
    
    if df is None:
        print("❌ No dataset available for analysis")
        return
    
    try:
        # Basic statistics for numerical columns
        print("\n" + "-"*40)
        print("BASIC STATISTICS (Numerical Columns):")
        print("-"*40)
        numerical_stats = df.describe()
        print(numerical_stats)
        
        # Additional statistics
        print("\n" + "-"*40)
        print("ADDITIONAL STATISTICS BY SPECIES:")
        print("-"*40)
        
        # Group by species and compute mean for each numerical column
        species_groups = df.groupby('species')
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if column != 'species':  # Skip the target column if it's numerical
                print(f"\nMean {column} by species:")
                species_means = species_groups[column].mean()
                print(species_means)
                
                # Find species with maximum and minimum values
                max_species = species_means.idxmax()
                min_species = species_means.idxmin()
                print(f"Highest {column}: {max_species} ({species_means[max_species]:.2f})")
                print(f"Lowest {column}: {min_species} ({species_means[min_species]:.2f})")
        
        # Interesting findings
        print("\n" + "-"*40)
        print("INTERESTING FINDINGS:")
        print("-"*40)
        
        # Correlation analysis
        numerical_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        
        print("\nCorrelation Matrix (Numerical Features):")
        print(correlation_matrix)
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = abs(correlation_matrix.iloc[i, j])
                if corr > 0.7:  # Strong correlation threshold
                    strong_correlations.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if strong_correlations:
            print("\nStrong Correlations (|r| > 0.7):")
            for col1, col2, corr in strong_correlations:
                print(f"{col1} vs {col2}: {corr:.3f}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

# Task 3: Data Visualization
def create_visualizations(df):
    """Task 3: Create various visualizations"""
    print("\n" + "="*50)
    print("TASK 3: DATA VISUALIZATION")
    print("="*50)
    
    if df is None:
        print("❌ No dataset available for visualization")
        return
    
    try:
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Visualization 1: Line chart (simulated time series)
        print("\n📈 Creating Visualization 1: Line Chart (Feature Trends by Species)")
        plt.subplot(2, 3, 1)
        
        # Simulate time series by using index as time
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            plt.plot(species_data.index[:30], species_data['sepal length (cm)'][:30], 
                    label=f'{species} sepal length', marker='o', linewidth=2)
        
        plt.title('Sepal Length Trends by Species (First 30 Samples)', fontsize=12, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Sepal Length (cm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Visualization 2: Bar chart (average measurements by species)
        print("📊 Creating Visualization 2: Bar Chart (Average Measurements by Species)")
        plt.subplot(2, 3, 2)
        
        species_means = df.groupby('species')[features].mean()
        species_means.plot(kind='bar', ax=plt.gca())
        plt.title('Average Measurements by Iris Species', fontsize=12, fontweight='bold')
        plt.xlabel('Species')
        plt.ylabel('Measurement (cm)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Visualization 3: Histogram (distribution of sepal length)
        print("📋 Creating Visualization 3: Histogram (Sepal Length Distribution)")
        plt.subplot(2, 3, 3)
        
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            plt.hist(species_data['sepal length (cm)'], alpha=0.7, label=species, bins=15)
        
        plt.title('Distribution of Sepal Length by Species', fontsize=12, fontweight='bold')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Visualization 4: Scatter plot (sepal length vs petal length)
        print("🔵 Creating Visualization 4: Scatter Plot (Sepal vs Petal Length)")
        plt.subplot(2, 3, 4)
        
        colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            plt.scatter(species_data['sepal length (cm)'], 
                       species_data['petal length (cm)'], 
                       c=colors[species], label=species, alpha=0.7, s=60)
        
        plt.title('Sepal Length vs Petal Length', fontsize=12, fontweight='bold')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Visualization 5: Box plot (distribution by species)
        print("📦 Creating Visualization 5: Box Plot (Feature Distribution by Species)")
        plt.subplot(2, 3, 5)
        
        df_melted = pd.melt(df, id_vars=['species'], value_vars=features)
        sns.boxplot(data=df_melted, x='variable', y='value', hue='species')
        plt.title('Distribution of Features by Species', fontsize=12, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Measurement (cm)')
        plt.xticks(rotation=45)
        
        # Visualization 6: Correlation heatmap
        print("🔥 Creating Visualization 6: Correlation Heatmap")
        plt.subplot(2, 3, 6)
        
        numerical_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Additional: Pairplot for comprehensive analysis
        print("\n🎨 Creating Additional Visualization: Pairplot")
        plt.figure(figsize=(12, 10))
        sns.pairplot(df, hue='species', palette='husl', diag_kind='hist', markers=['o', 's', 'D'])
        plt.suptitle('Iris Dataset Pairplot by Species', y=1.02, fontsize=14, fontweight='bold')
        plt.show()
        
        print("✅ All visualizations created successfully!")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")

def main():
    """Main function to run all tasks"""
    try:
        # Task 1: Load and explore dataset
        df = load_and_explore_dataset()
        
        if df is not None:
            # Task 2: Basic data analysis
            perform_basic_analysis(df)
            
            # Task 3: Data visualization
            create_visualizations(df)
            
            print("\n" + "="*60)
            print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Summary of findings
            print("\n📋 KEY FINDINGS SUMMARY:")
            print("- Iris dataset contains 150 samples of 3 species")
            print("- Setosa has distinctly smaller petals than other species")
            print("- Petal length and width show strong correlation")
            print("- Virginica has the largest measurements on average")
            print("- Species are well separated in feature space")
            
        else:
            print("❌ Project failed due to dataset loading issues")
            
    except KeyboardInterrupt:
        print("\n⚠️  Project execution interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# Run the main function
if __name__ == "__main__":
    main()