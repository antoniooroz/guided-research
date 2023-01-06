class SBM:
    def init(graph_data: GraphData, seed):
        import time
        start_time = time.time()
        
        adj_matrix = SBM.generate_graph(graph_data, seed)
        end_time_adj_matrix = time.time()
        print(f'adj_matrix: {end_time_adj_matrix-start_time}')
        
        features, labels = SBM.features(graph_data, seed)
        end_time_features = time.time()
        print(f'features: {end_time_features-end_time_adj_matrix}')
        
        graph = SparseGraph(
            adj_matrix=adj_matrix,
            attr_matrix=features,
            labels=labels
        )
        
        graph.standardize(select_lcc=True)
        graph.normalize_features(experiment_configuration=graph_data.experiment_configuration)
        
        return graph
    
    def generate_graph(graph_data: GraphData, seed):
        # Create graph
        networkx_graph = nx.stochastic_block_model(
            graph_data.experiment_configuration.sbm_classes,
            SBM.sbm_connection_probabilities(graph_data),
            nodelist=None,
            seed=seed,
            directed=False,
            selfloops=False,
            sparse=True
        )
        
        """https://networkx.org/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html"""
        
        """
        options = {"edgecolors": "tab:gray", "node_size": 25, "alpha": 1}
        
        plt.figure(figsize=(12, 10))
        spring_k =4/sqrt(sum(graph_data.experiment_configuration.sbm_classes)) # default k is 1/sqrt
        pos = nx.spring_layout(networkx_graph, seed=seed, scale=1, k=spring_k)
        
        i = 0
        colors = ["green", "blue", "cyan", "red"]
        for sbm_class, color in zip(graph_data.experiment_configuration.sbm_classes, colors):
            nx.draw_networkx_nodes(networkx_graph, pos, nodelist=range(i, i+sbm_class), node_color=f"tab:{color}", **options)
            i+=sbm_class
        
        nx.draw_networkx_edges(networkx_graph, pos, width=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{os.getcwd()}/plots/graphs/{seed}.png')
        plt.clf()
        """
        
        n_nodes = sum(graph_data.experiment_configuration.sbm_classes)
        
        adj_matrix=sp.sparse.csr_matrix(
            ([1]*len(networkx_graph.edges), (list(map(lambda x: x[0], networkx_graph.edges)), list(map(lambda x: x[1], networkx_graph.edges)))), shape=(n_nodes, n_nodes)
        )
        
        return adj_matrix
    
    def sbm_connection_probabilities(graph_data: GraphData):
        # TODO: For SBM also allow frac ood, etc.
        
        N = len(graph_data.experiment_configuration.sbm_classes)
        
        P_ID_IN = graph_data.experiment_configuration.sbm_connection_probabilities_id_in_cluster
        P_ID_OUT = graph_data.experiment_configuration.sbm_connection_probabilities_id_out_cluster
        P_OOD_IN = graph_data.experiment_configuration.sbm_connection_probabilities_ood_in_cluster
        P_OOD_OUT = graph_data.experiment_configuration.sbm_connection_probabilities_ood_out_cluster
        
        # ID
        connection_probabilities = (np.ones([N, N]) - np.eye(N)) * P_ID_OUT
        connection_probabilities += np.diag(np.ones([N]) * P_ID_IN)
        
        if graph_data.experiment_configuration.ood == OOD.LOC:
            OOD_N = graph_data.experiment_configuration.ood_loc_num_classes
        
            connection_probabilities[N-OOD_N:, :] = P_OOD_OUT
            connection_probabilities[:, N-OOD_N:] = P_OOD_OUT
        
            for i in range(N-OOD_N, N):
                connection_probabilities[i,i] = P_OOD_IN
        
        return connection_probabilities
        
    def features(graph_data: GraphData, seed):
        features = []
        labels = []
        
        random_state = seed
        
        means = []
        
        for c, nsamples in enumerate(graph_data.experiment_configuration.sbm_classes):
            mean = sp.stats.norm.rvs(loc=graph_data.experiment_configuration.sbm_feature_mean, scale=graph_data.experiment_configuration.sbm_feature_variance ,size=graph_data.experiment_configuration.sbm_nfeatures, random_state=random_state)
            means.append(mean)
            
            samples = sp.stats.multivariate_normal(
                mean=mean,
                cov=np.diag(np.abs(sp.stats.norm.rvs(loc=graph_data.experiment_configuration.sbm_feature_sampling_variance, scale=1, size=graph_data.experiment_configuration.sbm_nfeatures, random_state=random_state+1))),
                seed=seed
            ).rvs(size=nsamples)
            
            random_state += 10
            
            features.append(samples)
            labels.extend([c] * nsamples)
            
        features = np.concatenate(features)
        labels = np.asarray(labels)
            
        graph_data.log_feature_distances(means)
            
        return features, labels