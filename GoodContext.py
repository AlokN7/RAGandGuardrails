"""
Benign Context Generator
Path: RAGandGuardrailsFinal/fetch_clean_data.py
"""

import wikipedia
import os

def generate_clean_db():
    output_dir = "data_clean"
    os.makedirs(output_dir, exist_ok=True)

    topics = [
        # Academic & Computer Science
        "Design and analysis of algorithms", "Programming language", "Data structure", 
        "Distributed computing", "Software engineering", "Operating system", 
        "Compiler construction", "Graph theory", "Computer network",
        
        # Enterprise & Cybersecurity
        "Identity and access management", "Zero trust security model", "Information security",
        "Cryptography", "Public key infrastructure", "Network security", 
        "Risk management", "Regulatory compliance", "Information privacy",
        "Authentication", "Access control", "Data loss prevention software",
        
        # General Knowledge / Diverse Benign Context
        "Horology", "Mechanical watch", "Escapement", "Automatic watch",
        "Perfumery", "Essential oil", "Sandalwood", "Bergamot orange", "Vanilla",
        "Nuclear power", "Renewable energy", "Electric battery", "Energy storage",
        "Solar power", "Smart grid", "Wind power", "Geothermal energy",
        "Stock exchange", "Investment", "Video game industry", "Cloud computing",
        "Machine learning", "Natural language processing", "Artificial neural network"
    ]

    print(f"[*] Fetching {len(topics)} benign articles from Wikipedia...")
    
    success_count = 0
    for topic in topics:
        try:
            # Auto-suggest resolves slight naming variations
            page = wikipedia.page(topic, auto_suggest=True)
            
            # Clean the title for use as a filename
            safe_title = "".join([c for c in page.title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            file_path = os.path.join(output_dir, f"{safe_title.replace(' ', '_')}.txt")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(page.content)
                
            print(f"  [+] Downloaded: {page.title}")
            success_count += 1
            
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  [!] Skipping '{topic}': Disambiguation page.")
        except wikipedia.exceptions.PageError:
            print(f"  [!] Skipping '{topic}': Page not found.")
        except Exception as e:
            print(f"  [!] Error on '{topic}': {e}")

    print(f"\n✅ Successfully downloaded {success_count} benign documents into ./{output_dir}/")

if __name__ == "__main__":
    generate_clean_db()