import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from thefuzz import fuzz, process
from io import StringIO

class RecordLinker:
    def __init__(self):
        self.df_a = None
        self.df_b = None
        self.matched = None
        self.unmatched_a = None
        self.unmatched_b = None
        self.suspects = None
        self.rules = self.load_rules()
        
    def load_rules(self):
        try:
            with open('rule_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default rules if config doesn't exist
            return {
                "tiers": [
                    {
                        "name": "Exact Invoice ID Match",
                        "type": "exact",
                        "field_a": "invoice_id",
                        "field_b": "ref_code",
                        "active": True,
                        "weight": 1.0
                    },
                    {
                        "name": "Fuzzy Name Match",
                        "type": "fuzzy",
                        "field_a": "customer_name",
                        "field_b": "client",
                        "threshold": 90,
                        "active": True,
                        "weight": 0.8
                    },
                    {
                        "name": "Email Match",
                        "type": "exact",
                        "field_a": "customer_email",
                        "field_b": "email",
                        "active": True,
                        "weight": 0.9
                    },
                    {
                        "name": "Amount Match",
                        "type": "numeric",
                        "field_a": "total_amount",
                        "field_b": "grand_total",
                        "tolerance": 0.01,
                        "active": True,
                        "weight": 0.7
                    }
                ],
                "tie_breakers": [
                    {
                        "field": "invoice_date",
                        "direction": "latest"
                    },
                    {
                        "field": "total_amount",
                        "direction": "highest"
                    }
                ]
            }
    
    def save_rules(self):
        with open('rule_config.json', 'w') as f:
            json.dump(self.rules, f, indent=4)
    
    def preprocess_data(self):
        # Clean and standardize data
        for df in [self.df_a, self.df_b]:
            if df is not None:
                # Convert amount fields to numeric
                amount_cols = ['amount', 'tax_amount', 'total_amount', 'net', 'tax', 'grand_total']
                for col in amount_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Standardize date fields
                date_cols = ['invoice_date', 'doc_date']
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
                
                # Clean string fields
                string_cols = ['invoice_id', 'po_number', 'customer_name', 'customer_email', 
                              'ref_code', 'purchase_order', 'client', 'email']
                for col in string_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip().str.lower()
    
    def extract_invoice_number(self, value):
        """Extract numeric part from invoice ID for pattern matching"""
        if pd.isna(value):
            return None
        numbers = re.findall(r'\d+', str(value))
        return numbers[0] if numbers else None
    
    def apply_rules(self, record_a, record_b, rules):
        rationale = []
        total_score = 0
        max_score = sum(rule['weight'] for rule in rules if rule['active'])
        
        for rule in rules:
            if not rule['active']:
                continue
                
            field_a = rule.get('field_a')
            field_b = rule.get('field_b')
            
            if field_a not in record_a or field_b not in record_b:
                continue
                
            value_a = record_a[field_a]
            value_b = record_b[field_b]
            
            if pd.isna(value_a) or pd.isna(value_b):
                rationale.append(f"{rule['name']}: Missing values")
                continue
                
            if rule['type'] == 'exact':
                if value_a == value_b:
                    score = rule['weight']
                    total_score += score
                    rationale.append(f"{rule['name']}: Exact match ({value_a} == {value_b}) - Score: {score}")
                else:
                    rationale.append(f"{rule['name']}: No exact match ({value_a} != {value_b})")
            
            elif rule['type'] == 'fuzzy':
                similarity = fuzz.ratio(str(value_a), str(value_b))
                threshold = rule.get('threshold', 85)
                if similarity >= threshold:
                    score = rule['weight'] * (similarity / 100)
                    total_score += score
                    rationale.append(f"{rule['name']}: Fuzzy match ({similarity}% similar) - Score: {score:.2f}")
                else:
                    rationale.append(f"{rule['name']}: Insufficient similarity ({similarity}% < {threshold}%)")
            
            elif rule['type'] == 'numeric':
                tolerance = rule.get('tolerance', 0.05)
                if abs(float(value_a) - float(value_b)) <= tolerance * float(value_a):
                    score = rule['weight']
                    total_score += score
                    rationale.append(f"{rule['name']}: Numeric match within {tolerance*100}% tolerance - Score: {score}")
                else:
                    rationale.append(f"{rule['name']}: Numeric difference too large ({value_a} vs {value_b})")
            
            elif rule['type'] == 'pattern':
                # Handle patterned matches (e.g., invoice number extraction)
                pattern_a = self.extract_invoice_number(value_a)
                pattern_b = self.extract_invoice_number(value_b)
                
                if pattern_a and pattern_b and pattern_a == pattern_b:
                    score = rule['weight']
                    total_score += score
                    rationale.append(f"{rule['name']}: Pattern match ({pattern_a} found in both) - Score: {score}")
                else:
                    rationale.append(f"{rule['name']}: No pattern match")
        
        confidence = total_score / max_score if max_score > 0 else 0
        return confidence, rationale
    
    def resolve_ties(self, candidates, tie_breakers):
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]['record_b']
        
        # Apply tie-breakers in order
        for tie_breaker in tie_breakers:
            field = tie_breaker['field']
            direction = tie_breaker['direction']
            
            # Convert field names between datasets if needed
            if field == 'invoice_date':
                field_b = 'doc_date'
            elif field == 'total_amount':
                field_b = 'grand_total'
            else:
                field_b = field
                
            # Sort candidates based on tie-breaker
            if direction == 'latest':
                candidates.sort(key=lambda x: x['record_b'].get(field_b, ''), reverse=True)
            elif direction == 'earliest':
                candidates.sort(key=lambda x: x['record_b'].get(field_b, ''))
            elif direction == 'highest':
                candidates.sort(key=lambda x: x['record_b'].get(field_b, 0), reverse=True)
            elif direction == 'lowest':
                candidates.sort(key=lambda x: x['record_b'].get(field_b, 0))
            
            # Return the best candidate according to this tie-breaker
            return candidates[0]['record_b']
        
        # If no tie-breakers resolve it, return the first candidate
        return candidates[0]['record_b']
    
    def link_records(self):
        if self.df_a is None or self.df_b is None:
            st.error("Please upload both datasets first.")
            return
        
        self.preprocess_data()
        
        active_rules = [rule for rule in self.rules['tiers'] if rule['active']]
        tie_breakers = self.rules.get('tie_breakers', [])
        
        matched_records = []
        unmatched_a = self.df_a.copy()
        unmatched_b = self.df_b.copy()
        suspect_records = []
        
        # For each record in source A, find the best match in source B
        for idx_a, record_a in self.df_a.iterrows():
            candidates = []
            
            for idx_b, record_b in self.df_b.iterrows():
                confidence, rationale = self.apply_rules(record_a, record_b, active_rules)
                
                if confidence > 0:
                    candidates.append({
                        'record_b': record_b,
                        'confidence': confidence,
                        'rationale': rationale
                    })
            
            if candidates:
                # Find the best candidate
                best_candidate = max(candidates, key=lambda x: x['confidence'])
                
                if best_candidate['confidence'] >= 0.8:
                    # Confident match
                    matched_records.append({
                        'record_a': record_a,
                        'record_b': best_candidate['record_b'],
                        'confidence': best_candidate['confidence'],
                        'rationale': best_candidate['rationale'],
                        'status': 'matched'
                    })
                    
                    # Remove from unmatched
                    unmatched_b = unmatched_b[unmatched_b.index != best_candidate['record_b'].name]
                else:
                    # Suspect match (low confidence)
                    suspect_records.append({
                        'record_a': record_a,
                        'record_b': best_candidate['record_b'],
                        'confidence': best_candidate['confidence'],
                        'rationale': best_candidate['rationale'],
                        'status': 'suspect'
                    })
            else:
                # No match found
                matched_records.append({
                    'record_a': record_a,
                    'record_b': None,
                    'confidence': 0,
                    'rationale': ["No matching rules triggered"],
                    'status': 'unmatched'
                })
        
        # Prepare results
        self.matched = pd.DataFrame([{
            **{f"a_{col}": r['record_a'][col] for col in self.df_a.columns},
            **{f"b_{col}": r['record_b'][col] if r['record_b'] is not None else None for col in self.df_b.columns},
            'confidence': r['confidence'],
            'status': r['status'],
            'rationale': " | ".join(r['rationale'])
        } for r in matched_records])
        
        self.unmatched_a = self.df_a[~self.df_a.index.isin([r['record_a'].name for r in matched_records if r['status'] == 'matched'])]
        self.unmatched_b = unmatched_b
        
        self.suspects = pd.DataFrame([{
            **{f"a_{col}": r['record_a'][col] for col in self.df_a.columns},
            **{f"b_{col}": r['record_b'][col] for col in self.df_b.columns},
            'confidence': r['confidence'],
            'rationale': " | ".join(r['rationale'])
        } for r in suspect_records])
        
        return True

def main():
    st.set_page_config(page_title="Cross-Source Record Linker", layout="wide")
    st.title("Cross-Source Record Linking")
    
    if 'linker' not in st.session_state:
        st.session_state.linker = RecordLinker()
    
    linker = st.session_state.linker
    
    # File upload
    st.sidebar.header("Data Upload")
    file_a = st.sidebar.file_uploader("Upload Source A CSV", type=['csv'])
    file_b = st.sidebar.file_uploader("Upload Source B CSV", type=['csv'])
    
    if file_a and file_b:
        linker.df_a = pd.read_csv(file_a)
        linker.df_b = pd.read_csv(file_b)
        
        st.sidebar.success("Data uploaded successfully!")
        
        # Rule configuration
        st.sidebar.header("Rule Configuration")
        
        for i, rule in enumerate(linker.rules['tiers']):
            with st.sidebar.expander(f"Rule: {rule['name']}"):
                rule['active'] = st.checkbox("Active", value=rule['active'], key=f"active_{i}")
                
                if rule['type'] == 'fuzzy':
                    rule['threshold'] = st.slider("Similarity Threshold", 0, 100, 
                                                 value=rule.get('threshold', 85), 
                                                 key=f"threshold_{i}")
                elif rule['type'] == 'numeric':
                    rule['tolerance'] = st.slider("Tolerance (%)", 0.0, 1.0, 
                                                 value=rule.get('tolerance', 0.05), 
                                                 step=0.01, key=f"tolerance_{i}")
                
                rule['weight'] = st.slider("Weight", 0.0, 1.0, 
                                          value=rule.get('weight', 0.7), 
                                          step=0.1, key=f"weight_{i}")
        
        # Tie-breaker configuration
        st.sidebar.header("Tie-Breaker Settings")
        for i, tb in enumerate(linker.rules['tie_breakers']):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                tb['field'] = st.selectbox(
                    "Field", 
                    options=['invoice_date', 'total_amount', 'amount', 'tax_amount'],
                    index=['invoice_date', 'total_amount', 'amount', 'tax_amount'].index(tb['field']),
                    key=f"tb_field_{i}"
                )
            with col2:
                tb['direction'] = st.selectbox(
                    "Direction",
                    options=['latest', 'earliest', 'highest', 'lowest'],
                    index=['latest', 'earliest', 'highest', 'lowest'].index(tb['direction']),
                    key=f"tb_dir_{i}"
                )
        
        if st.sidebar.button("Save Configuration"):
            linker.save_rules()
            st.sidebar.success("Configuration saved!")
        
        # Run matching
        if st.sidebar.button("Run Record Linking"):
            with st.spinner("Linking records..."):
                success = linker.link_records()
            
            if success:
                st.sidebar.success("Record linking completed!")
    
    # Display results
    if linker.matched is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["Matched", "Suspect", "Unmatched A", "Unmatched B"])
        
        with tab1:
            st.header("Matched Records")
            st.dataframe(linker.matched[linker.matched['status'] == 'matched'].drop(columns=['status']))
            
            # Show match statistics
            matched_count = len(linker.matched[linker.matched['status'] == 'matched'])
            total_count = len(linker.df_a)
            st.metric("Match Rate", f"{(matched_count/total_count*100):.1f}%", 
                     f"{matched_count} of {total_count} records")
        
        with tab2:
            st.header("Suspect Matches")
            if len(linker.suspects) > 0:
                st.dataframe(linker.suspects)
                
                # Pattern adoption interface
                st.subheader("Adopt New Pattern")
                selected_idx = st.selectbox("Select suspect match to analyze", range(len(linker.suspects)))
                
                if selected_idx is not None:
                    suspect = linker.suspects.iloc[selected_idx]
                    st.write("**Rationale:**", suspect['rationale'])
                    
                    # Pattern suggestion
                    a_invoice = suspect['a_invoice_id']
                    b_ref = suspect['b_ref_code']
                    
                    # Try to find a pattern
                    a_numbers = re.findall(r'\d+', a_invoice)
                    b_numbers = re.findall(r'\d+', b_ref)
                    
                    if a_numbers and b_numbers and a_numbers[0] == b_numbers[0]:
                        st.success(f"Pattern detected: Both IDs contain {a_numbers[0]}")
                        
                        if st.button("Adopt This Pattern as New Rule"):
                            # Add new pattern rule
                            new_rule = {
                                "name": f"Pattern Match: {a_numbers[0]} in IDs",
                                "type": "pattern",
                                "field_a": "invoice_id",
                                "field_b": "ref_code",
                                "active": True,
                                "weight": 0.7
                            }
                            linker.rules['tiers'].append(new_rule)
                            linker.save_rules()
                            st.success("New pattern rule added! Run matching again to apply.")
            else:
                st.info("No suspect matches found.")
        
        with tab3:
            st.header("Unmatched Records from Source A")
            st.dataframe(linker.unmatched_a)
        
        with tab4:
            st.header("Unmatched Records from Source B")
            st.dataframe(linker.unmatched_b)
        
        # Export results
        st.sidebar.header("Export Results")
        if st.sidebar.button("Export All Results as CSV"):
            csv = linker.matched.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name="record_linking_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Upload both datasets and run record linking to see results.")

if __name__ == "__main__":
    main()