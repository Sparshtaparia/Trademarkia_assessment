import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Semantic Search System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .cache-hit {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .cache-miss {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .query-result {
        background: #f8f9fa;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 10px 0;
    }
    .cluster-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 5px 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "cache_stats" not in st.session_state:
    st.session_state.cache_stats = None

# Sidebar
st.sidebar.title("⚙️ Configuration")

api_url = st.sidebar.text_input(
    "API Base URL",
    value=API_BASE_URL,
    help="URL of the FastAPI backend"
)

top_k = st.sidebar.slider(
    "Top K Results",
    min_value=1,
    max_value=20,
    value=5,
    help="Number of search results to return"
)

similarity_threshold = st.sidebar.slider(
    "Cache Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.05,
    help="Minimum similarity score for cache hits"
)

# Check API health
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 System Status")

try:
    response = requests.get(f"{api_url}/health", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✓ API Connected")
    else:
        st.sidebar.error("✗ API Error")
except:
    st.sidebar.error("✗ Cannot reach API")

# Main content
st.title("🔍 Semantic Search System")
st.markdown("""
Powered by Sentence Transformers, Gaussian Mixture Clustering, and Pinecone Vector Database
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Search", "Cache Analytics", "Clustering Insights", "Query History"])

# ============ TAB 1: SEARCH ============
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_text = st.text_area(
            "Enter your search query",
            placeholder="e.g., machine learning, deep neural networks, artificial intelligence...",
            height=100,
            label_visibility="collapsed",
            key="search_query_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True, key="search_btn")
        clear_cache_btn = st.button("🗑️ Clear Cache", use_container_width=True)
    
    if clear_cache_btn:
        try:
            response = requests.delete(f"{api_url}/cache")
            if response.status_code == 200:
                st.success("Cache cleared successfully!")
                st.session_state.query_history = []
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
    
    if search_button and query_text.strip():
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{api_url}/query",
                    json={
                        "text": query_text,
                        "top_k": top_k,
                        "cache_threshold": similarity_threshold
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store in history
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": query_text,
                        "result": result
                    })
                    
                    # Display cache status
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if result.get("cache_hit"):
                            st.markdown("""
                            <div class="metric-card cache-hit">
                                <div style="font-size: 14px; opacity: 0.9;">Cache Status</div>
                                <div style="font-size: 24px; font-weight: bold;">🎯 HIT</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="metric-card cache-miss">
                                <div style="font-size: 14px; opacity: 0.9;">Cache Status</div>
                                <div style="font-size: 24px; font-weight: bold;">❌ MISS</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        similarity = result.get("max_similarity_score", 0)
                        st.metric("Similarity Score", f"{similarity:.3f}", delta=f"{similarity*100:.1f}%")
                    
                    with col3:
                        st.metric("Dominant Cluster", result.get("dominant_cluster", "N/A"))
                    
                    with col4:
                        st.metric("Results Found", len(result.get("results", [])))
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📄 Search Results")
                    
                    for idx, item in enumerate(result.get("results", []), 1):
                        with st.expander(f"#{idx} - {item.get('title', 'Document')[:60]}...", expanded=idx<=2):
                            st.markdown(f"**Score:** {item.get('score', 0):.4f}")
                            st.markdown(f"**Cluster:** {item.get('cluster_id', 'N/A')}")
                            st.text_area(
                                "Content",
                                value=item.get('content', '')[:500] + "...",
                                disabled=True,
                                height=150,
                                label_visibility="collapsed",
                                key=f"result_content_{idx}"
                            )
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============ TAB 2: CACHE ANALYTICS ============
with tab2:
    if st.button("📊 Refresh Cache Stats", key="refresh_stats"):
        with st.spinner("Loading cache statistics..."):
            try:
                response = requests.get(f"{api_url}/cache/stats")
                if response.status_code == 200:
                    st.session_state.cache_stats = response.json()
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")
    
    if st.session_state.cache_stats:
        stats = st.session_state.cache_stats
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries",
                stats.get("total_queries", 0),
                delta=stats.get("total_queries_diff", 0)
            )
        
        with col2:
            hit_rate = (stats.get("cache_hits", 0) / max(stats.get("total_queries", 1), 1)) * 100
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        
        with col3:
            st.metric("Cache Hits", stats.get("cache_hits", 0))
        
        with col4:
            st.metric("Cache Misses", stats.get("cache_misses", 0))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Hit vs Miss pie chart
            fig = go.Figure(data=[go.Pie(
                labels=["Hits", "Misses"],
                values=[stats.get("cache_hits", 0), stats.get("cache_misses", 0)],
                marker=dict(colors=["#f093fb", "#4facfe"]),
                hole=0.3
            )])
            fig.update_layout(title="Cache Hit/Miss Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cluster distribution
            cluster_dist = stats.get("cluster_distribution", {})
            if cluster_dist:
                fig = go.Figure(data=[go.Bar(
                    x=list(cluster_dist.keys()),
                    y=list(cluster_dist.values()),
                    marker=dict(color="#667eea")
                )])
                fig.update_layout(
                    title="Queries per Cluster",
                    xaxis_title="Cluster ID",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Cache entries by cluster
        st.subheader("📦 Cache Entries by Cluster")
        cache_by_cluster = stats.get("cache_entries_by_cluster", {})
        if cache_by_cluster:
            df = pd.DataFrame([
                {"Cluster": k, "Cached Entries": v}
                for k, v in sorted(cache_by_cluster.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No cache entries yet")
    else:
        st.info("Click 'Refresh Cache Stats' to load analytics")

# ============ TAB 3: CLUSTERING INSIGHTS ============
with tab3:
    st.subheader("📊 Clustering Analysis")
    st.markdown("""
    This system uses **Gaussian Mixture Model (GMM)** for soft clustering.
    Each document can belong to multiple clusters with different probabilities.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Soft Clustering
        - Probabilistic cluster assignments
        - Documents can belong to multiple clusters
        - Cluster probabilities indicate confidence
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Optimization Metrics
        - **BIC Score**: Model complexity vs fit
        - **Silhouette Score**: Cluster cohesion
        - **Davies-Bouldin Index**: Cluster separation
        """)
    
    st.markdown("---")
    st.subheader("📈 Sample Clustering Results")
    
    sample_data = {
        "Document": ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"],
        "Primary Cluster": [0, 1, 2, 0, 3],
        "Primary Probability": [0.85, 0.92, 0.78, 0.88, 0.81],
        "Secondary Cluster": [1, 0, 1, 2, 0],
        "Secondary Probability": [0.12, 0.06, 0.18, 0.09, 0.15],
    }
    
    df_sample = pd.DataFrame(sample_data)
    st.dataframe(df_sample, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🔍 Cluster Keywords")
    
    col1, col2, col3, col4 = st.columns(4)
    
    clusters_keywords = {
        "0": ["machine", "learning", "neural", "deep", "training"],
        "1": ["data", "analysis", "statistics", "visualization", "mining"],
        "2": ["computer", "vision", "image", "detection", "recognition"],
        "3": ["natural", "language", "processing", "nlp", "text"]
    }
    
    cols = st.columns(4)
    
    for idx, (cluster_id, keywords) in enumerate(clusters_keywords.items()):
        with cols[idx]:
            st.markdown(f"**Cluster {cluster_id}**")
            for kw in keywords:
                st.caption(f"• {kw}")

# ============ TAB 4: QUERY HISTORY ============
with tab4:
    st.subheader("📋 Query History")
    
    if st.session_state.query_history:
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
        
        st.markdown("---")
        
        for idx, item in enumerate(reversed(st.session_state.query_history), 1):
            with st.expander(f"Query {len(st.session_state.query_history) - idx + 1}: {item['query'][:60]}..."):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.text_area(
                        "Query Text",
                        value=item["query"],
                        disabled=True,
                        height=80,
                        label_visibility="collapsed",
                        key=f"history_query_{idx}"
                    )
                
                with col2:
                    result = item["result"]
                    st.metric("Cache Hit", "✓ Yes" if result.get("cache_hit") else "✗ No")
                    st.metric("Score", f"{result.get('max_similarity_score', 0):.3f}")
                    st.metric("Results", len(result.get("results", [])))
                
                st.caption(f"Executed at: {item['timestamp']}")
                
                if st.button("📤 Export", key=f"export_{idx}"):
                    st.download_button(
                        label="Download as JSON",
                        data=json.dumps(item, indent=2),
                        file_name=f"query_{idx}.json",
                        mime="application/json"
                    )
    else:
        st.info("No queries executed yet. Start by searching in the 'Search' tab!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999;">
    <small>Semantic Search System | Powered by Sentence Transformers, Pinecone, and Streamlit</small>
</div>
""", unsafe_allow_html=True)
