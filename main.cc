#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stack>
#include <format>
#include <random>

#define LOCALTEST
using namespace std;

const int INF = 1e9;
const int NONE = -1;
const int INITIAL_DIST = 10000;
vector<uint32_t> HASHTABLE;

void print(vector<int> &vec)
{
	for (auto a : vec)
		cerr << a << " ";
	cerr << endl;
}

class logHandlerClass
{
private:
	std::vector<std::string> buffer;

	// 可変引数がない場合の処理
	void format_helper(std::ostringstream &oss, const std::string &format)
	{
		oss << format;
	}

	template <typename T, typename... Args>
	void format_helper(std::ostringstream &oss, const std::string &format, T value, Args... args)
	{
		size_t pos = format.find("{}");
		if (pos != std::string::npos)
		{
			oss << format.substr(0, pos) << value;
			format_helper(oss, format.substr(pos + 2), args...);
		}
		else
		{
			oss << format;
		}
	}

	template <typename... Args>
	std::string format(const std::string &format, Args... args)
	{
		std::ostringstream oss;
		format_helper(oss, format, args...);
		return oss.str();
	}

public:
	template <typename... Args>
	void log(const std::string fmt, Args... args)
	{
		auto message = format(fmt, args...);
		buffer.push_back(message);
	}

	void flush(std::ostream &os = std::cerr)
	{
		for (const auto &msg : buffer)
		{
			os << msg << std::endl;
		}
		buffer.clear();
	}

	void clear() { buffer.clear(); }
};

/* timer class */
class timerClass
{
private:
	std::chrono::system_clock::time_point mBegin;

public:
	timerClass() = default;
	inline std::chrono::system_clock::time_point now() const { return std::chrono::system_clock::now(); }
	void start() { mBegin = now(); }
	double elapsed() { return std::chrono::duration_cast<std::chrono::nanoseconds>(now() - mBegin).count() * 1.0e-9; }
};

timerClass timer1, timer2, timer3;
double e1 = 0, e2 = 0, e3 = 0;

class UnionFinding
{
public:
	UnionFinding() {}
	UnionFinding(int nsize)
	{
		par.reserve(nsize);
		for (int i = 0; i < nsize; ++i)
		{
			par.push_back(i);
		}
	}

	vector<int> par;

	void clear()
	{
		for (int i = 0; i < par.size(); ++i)
			par[i] = i;
	}

	bool Union(int x, int y)
	{
		x = root(x);
		y = root(y);
		// y = par[y];
		par[y] = x;
		return true;
	}

	bool Find(int x, int y)
	{
		return root(x) == root(y);
	}

	int root(int x)
	{
		if (par[x] == x)
			return x;
		par[x] = root(par[x]);
		return par[x];
	}
};

static unsigned long xor128()
{
	static unsigned long x = 123456789, y = 362436069, z = 521288629, w = 88675123;
	unsigned long t;
	t = (x ^ (x << 11));
	x = y;
	y = z;
	z = w;
	return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

uint32_t generate_random_uint32() {
    return static_cast<uint32_t>(xor128());
}

template <typename T>
void shuffleVector(std::vector<T>& container) {
    // ランダムデバイスとメルセンヌ・ツイスターエンジンを用意
    std::random_device rd; // ランダムデバイス (シード生成器)
    std::mt19937 g(rd());  // メルセンヌ・ツイスター19937エンジン

    // シャッフル
    std::shuffle(container.begin()+1, container.end(), g);
}

double rnd()
{
	return (double)xor128() / ULONG_MAX;
}

class posClass
{
private:
	using data_type = int;

public:
	posClass() = default;
	posClass(int id) : id(id) {}
	int id, done, pre;
	data_type cost;
};

class edgeClass
{
private:
	using data_type = int;

public:
	edgeClass() = default;
	edgeClass(string edge_name, int from, int to, data_type dist) : edge_name(edge_name), from(from), to(to), dist(dist) {}
	string edge_name;
	int from, to;
	data_type dist = 0;
	bool operator<(const edgeClass &other) const
	{
		return dist < other.dist;
	}
	bool operator>(const edgeClass &other) const
	{
		return dist > other.dist;
	}
};

template <typename T>
struct RadixHeap
{
	using uint = unsigned long long;
	vector<pair<uint, T>> v[33];
	uint size, last;

	RadixHeap() : size(0), last(0) {}

	bool empty() const { return size == 0; }
	inline int getbit(int a) { return a ? 32 - __builtin_clz(a) : 0; }
	void push(uint key, const T &value)
	{
		++size;
		v[getbit(key ^ last)].emplace_back(key, value);
	}

	pair<uint, T> pop()
	{
		if (v[0].empty())
		{
			int idx = 1;
			while (v[idx].empty())
				++idx;
			last = min_element(begin(v[idx]), end(v[idx]))->first;
			for (auto &p : v[idx])
				v[getbit(p.first ^ last)].emplace_back(p);
			v[idx].clear();
		}
		--size;
		auto ret = v[0].back();
		v[0].pop_back();
		return ret;
	}
};

class graphClass
{
private:
	using data_type = int;
	using P = pair<data_type, int>;
	using heap_type = RadixHeap<int>;

public:
	graphClass() = default;
	graphClass(int V, vector<posClass> &pos_vec) : V(V), pos_vec(pos_vec)
	{
		all_edge.resize(V, vector<edgeClass>(V));
		edge_vec.resize(V);
	}

	int V, E;
	vector<vector<edgeClass>> all_edge;
	vector<vector<edgeClass *>> edge_vec;
	vector<posClass> pos_vec;
	vector<vector<int>> pre_routes, sols;
	vector<vector<vector<int>>> routes;

	void add_edge(int u, int v, data_type dist)
	{
		all_edge[u][v] = {"", u, v, dist};
		all_edge[v][u] = {"", v, u, dist};

		edge_vec[u].push_back(&all_edge[u][v]);
		edge_vec[v].push_back(&all_edge[v][u]);
	}

	void init_dijkstra()
	{
		pre_routes.resize(V);
		sols.resize(V);
		routes.resize(V);

		for (auto &vec : pre_routes)
			vec.resize(V);
		for (auto &vec : sols)
			vec.resize(V, 0);
		for (auto &vec : routes)
			vec.resize(V);
	}

	int dijkstra(int from)
	{
		heap_type que;
		graphClass *g = this;

		const int n = g->V;
		for (auto &node : g->pos_vec)
		{
			node.done = false;
			node.cost = 1e8;
			node.pre = -1;
		}

		g->pos_vec[from].cost = 0;

		que.push(0, from);

		while (!que.empty())
		{
			P p = que.pop();
			int v = p.second;
			auto *sNode = &g->pos_vec[v];
			if (sNode->cost < p.first)
				continue;

			auto &edges = g->edge_vec[v];
			for (int i = 0; i < edges.size(); ++i)
			{
				int to = edges[i]->to;
				data_type dist = edges[i]->dist;
				auto *eNode = &g->pos_vec[to];
				if (eNode->cost > sNode->cost + dist)
				{
					eNode->cost = sNode->cost + dist;
					que.push(eNode->cost, to);
					eNode->pre = sNode->id;
				}
			}
		}

		auto &pre = pre_routes[from];
		auto &sol = sols[from];
		auto &rst = routes[from];

		for (int i = 0; i < n; ++i)
		{
			pre[i] = g->pos_vec[i].pre;
			sol[i] = g->pos_vec[i].cost;
		}
		for (int i = 0; i < n; ++i)
			rst[i] = getRoute(from, i);

		return 0;
	}

	int dijkstra()
	{
		init_dijkstra();
		for (int i = 0; i < V; ++i)
			dijkstra(i);
		return 0;
	}

	int getdist(int from, int to) { return from == to ? 10000000 : sols[from][to]; }

	vector<int> getRoute(int from, int to)
	{
		if (routes[from].size() > 0 && routes[from][to].size() > 0)
			return routes[from][to];
		vector<int> ret;
		ret.push_back(to);
		while (to != from && to >= 0)
		{
			to = pre_routes[from][to];
			if (0 <= to)
				ret.push_back(to);
		}
		reverse(ret.begin(), ret.end());
		return ret;
	}
};

class signalAClass
{
private:
	int pos_size;

public:
	signalAClass() = default;
	signalAClass(int pos_size, int signal_size) : pos_size(pos_size)
	{
		lists.resize(signal_size, NONE);
		exists = 0;
	}
	vector<int> lists;
	// vector<int> exists;
	bitset<600> exists;
	vector<vector<int>> indices;
	void mapping()
	{
		indices.clear();
		indices.resize(pos_size);
		for (int i = 0; i < lists.size(); ++i)
			if (lists[i] != NONE)
				indices[lists[i]].push_back(i);
		return;
	}
};

class signalBClass
{
private:
	// int pos_size;
	// int signal_size;
public:
	signalBClass() = default;
	signalBClass(int pos_size, int signal_size)
	{ // pos_size(pos_size), signal_size(signal_size) {
		lists.resize(signal_size, NONE);
		exists = 0;
		// exists.resize(pos_size, false);
	}
	vector<int> lists;
	bitset<600> exists;
};

class inputClass
{
public:
	inputClass() = default;
	int N, M, T, LA, LB;
	vector<int> dest_pos;
	vector<posClass> pos;
	graphClass graph;

	void input()
	{
		cin >> N >> M >> T >> LA >> LB;

		for (int i = 0; i < N; ++i)
			pos.push_back(posClass(i));
		graph = graphClass(N, pos);

		for (int m = 0; m < M; ++m)
		{
			int u, v;
			cin >> u >> v;
			graph.add_edge(u, v, INITIAL_DIST);
		}
		for (int t = 0; t < T; ++t)
		{
			int ti;
			cin >> ti;
			dest_pos.push_back(ti);
		}

		return;
	}
};

class actionClass
{
public:
	using Int = long long int;
private:
	Int field;
public:
	actionClass() = default;
	actionClass(Int left, Int right, Int index, Int direct, Int sig) { encode(left, right, index, direct, sig); }
	void encode(Int left, Int right, Int index, Int direct, Int sig) { field = left | (right << 11ll) | (index << 22ll) | (direct << 33ll) | (sig << 44ll);	}
	tuple<int, int, int, int, int> decode() { return {field & 2047ull, (field >> 11ll) & 2047ll, (field >> 22ll) & 2047ll, (field >> 33ll) & 2047ll, (field >> 44ll)}; }
};

class evaluatorClass
{
private:
	using Int = int;
	//int n_achievements, dist;
	Int field;
public:
	evaluatorClass() = default;
	evaluatorClass(Int dist1, Int dist2, Int n_achievements) { encode(dist1, dist2, n_achievements); }
	void encode(Int dist1, Int dist2, Int n_achievements) { field = (128 - (Int) dist1 / INITIAL_DIST) | ((128 - (Int) dist2 / INITIAL_DIST) << 7) | (n_achievements << 14); }
	tuple<int, int, int> decode() { return {(128 - field & 127) * INITIAL_DIST, (128 - (field >> 7) & 127) * INITIAL_DIST, (field >> 14)}; } 
	bool operator < (const evaluatorClass &other) const { return field < other.field; }
	bool operator > (const evaluatorClass &other) const { return field > other.field; }
};

class candidateClass
{
public:
	candidateClass() = default;
	int parent;
	actionClass action;
	evaluatorClass evaluator;
	bool operator<(const candidateClass &other) const
	{
		return evaluator < other.evaluator;
	}
	bool operator>(const candidateClass &other) const
	{
		return evaluator > other.evaluator;
	}
};

class temporaryHelperClass
{
private:
public:
	temporaryHelperClass() = default;
	temporaryHelperClass(int size) { signal.resize(size); }
	vector<int> signal;
	actionClass action;
	evaluatorClass evaluator;
	bitset<600> exists;
	int id, rank;
	
	uint32_t hash(const vector<int> &signal)
	{
		uint32_t ret = 0;
		for(auto e : signal) if(e != NONE) ret ^= HASHTABLE[e+1];
		return ret;
	}

	void set(int left, int right, int index, int direct, int sig, int id_, int rank_)
	{
		action = {left, right, index, direct, sig};
		id = id_; rank = rank_;
	}
	const vector<int> &new_signal(signalAClass &signalA, signalBClass &signalB)
	{
		
		exists = 0;
		//signal.clear();
		//signal.assign(signal.size(), NONE);
		for(auto &s : signal) s = NONE;

		auto [left, right, index, a, _] = action.decode();

		int j = 0;
		auto &listB = signalB.lists;
		auto &listA = signalA.lists;
		const int size = signalB.lists.size();
		for (int i = 0; i < size; ++i)
		{
			if (left <= i && i < right && a ==0)
				signal[j++] = index + i - right + 1;
			else if (left <= i && i < right && a == 1)
				signal[j++] = index + i - left;
			else
				signal[j++] = listB[i];	
		}
		
		for (auto a : signal)
			if (a != NONE)
				exists[listA[a]] = 1;
		
		return signal;
	}
};

class selectorClass
{
private:
public:
	selectorClass() = default;
	selectorClass(int LB)
	{
		uf = UnionFinding(LB);
	}

	unordered_map<uint32_t, int> hashMap;
	vector<candidateClass> candidates, candidates_all;
	UnionFinding uf;

	void clear()
	{
		candidates_all.clear();
		candidates.clear();
		hashMap.clear();
	}
		
	evaluatorClass estimate(int rank, int left, int right, signalAClass &signalA, inputClass &input)
	{
		auto &graph = input.graph;
		auto &destinations  = input.dest_pos;
		
		int n_a = rank, n_a2 = rank + 1 < input.N ? rank + 1 : rank;
		int dist_min = INF, dist_min_next = INF;

		do
		{
			int dest = destinations[n_a], dest2 = destinations[n_a2];
			dist_min = INF;
			dist_min_next = INF;

			for(int index = left; index <= right; ++index)
			{
				int a = signalA.lists[index];
				if (a != NONE && dist_min > graph.sols[dest][a])
					dist_min = input.graph.sols[dest][a];
				if (a != NONE && dist_min_next > graph.sols[dest2][a])
					dist_min_next = input.graph.sols[dest2][a];
					
			}
			if (dist_min == 0)
				n_a += 1;
		} while (dist_min == 0);

		return {dist_min_next, dist_min, n_a};
	}

	void push(inputClass &input, signalAClass &signalA, signalBClass &signalB, temporaryHelperClass &tempHelper)
	{
		candidateClass candidate;
		candidate.parent    = tempHelper.id;
		candidate.action    = tempHelper.action;
		candidate.evaluator = tempHelper.evaluator;
		candidates_all.push_back(std::move(candidate));
		
		return;
	}

	bool valid(signalAClass &signalA, signalBClass &signalB, graphClass &graph, temporaryHelperClass &tempHelper)
	{
		int ret = true;
		const vector<int> &p = tempHelper.new_signal(signalA, signalB);
		auto hash = tempHelper.hash(p);
		
		if(hashMap[hash]) return false;
		hashMap[hash] = 1;
		
		uf.clear();
		for (int i = 0; i < p.size(); ++i)
		{
			for (int j = i + 1; j < p.size(); ++j)
			{
				if(p[i] == NONE or p[j] == NONE) continue;
				int from = signalA.lists[p[i]], to = signalA.lists[p[j]];
				if (graph.all_edge[from][to].dist > 0)
				{
					uf.Union(i, j);
				}
			}
		}

		for (int i = 1; i < signalB.lists.size(); ++i)
			if (uf.root(i) != uf.root(0) && p[i] >= 0)
				ret = false;
		
		return ret;
	}

	void accept(signalAClass &signalA, signalBClass &signalB, graphClass &graph, temporaryHelperClass &tempHelper, candidateClass &candidate)
	{
		tempHelper.action = candidate.action;
		if(!valid(signalA, signalB, graph, tempHelper)) 
			return;
		// auto bitsetHash = bitsetHasher(tempHelper.exists);
		// hashMap[bitsetHash] = 1;
		candidates.push_back(candidate);
	}
};

class stateClass
{
private:

public:
	stateClass() = default;
	stateClass(signalBClass &signal) : signal(signal) {}
	int id = 0, parent = -1, rank = 0;
	int now_dist = 63 * INITIAL_DIST;
	signalBClass signal;

	void transition(int new_id, inputClass &input, signalAClass &signalA, temporaryHelperClass &tempHelper, candidateClass &candidate)
	{
		tempHelper.action = candidate.action;
		signal.lists = tempHelper.new_signal(signalA, signal);
		signal.exists = tempHelper.exists;
		id = new_id;
		auto [dist2, dist, n_achievements] = candidate.evaluator.decode();
		// rank = candidate.evaluator.n_achievements;
		// now_dist = candidate.evaluator.dist;
		rank = n_achievements;
		now_dist = dist;
		return;
	}

	void expand(inputClass &input, signalAClass &signalA, temporaryHelperClass &tempHelper, selectorClass &selector)
	{
		auto &graph = input.graph;
		//auto &dest  = input.dest_pos;

		bool initial_condition = true;
		for (auto &sig : signal.lists)
			if (sig != NONE)
				initial_condition = false;
		const int LB = signal.lists.size();
		if (initial_condition)
			signal.lists[0] = 0;

		evaluatorClass now_evaluator = {now_dist, now_dist, rank};

		for (auto &sig : signal.lists)
		{
			if (sig == NONE)
				continue;
			for (auto &edge : graph.edge_vec[signalA.lists[sig]])
			{
				vector<int> &indice = signalA.indices[edge->to];

				for (auto &index : indice)
				{
					if (signal.exists[signalA.lists[index]])
						continue;
					for (int direct = 0; direct <= 1; ++direct)
					{
						for (int j = (int)0.5 * LB; j <= LB; ++j)
						{
							if(direct == 0 && index-j+1 < 0) break;
							else if(direct == 1 && index + j - 1 >= input.LA) break;
							if(direct == 0) 
								tempHelper.evaluator = selector.estimate(rank, index-j+1, index, signalA, input);
							else 
								tempHelper.evaluator = selector.estimate(rank, index, index + j - 1, signalA, input);

							if(tempHelper.evaluator < now_evaluator)
								continue;

							for (int pos = 0; pos <= LB - j; ++pos)
							{
								int left = pos, right = pos + j;
								tempHelper.set(left, right, index, direct, sig, id, rank);
								selector.push(input, signalA, signal, tempHelper);

								if (initial_condition)
									break;
							}
						}
					}
				}
			}
		}
		if (initial_condition)
			signal.lists[0] = NONE;

		return;
	}

};

class solverClass
{
private:
	signalAClass signalA;
	signalBClass signalB;
	temporaryHelperClass tempHelper;
	inputClass input;
	vector<vector<candidateClass>> nodes;

public:
	solverClass()
	{
		input.input();

		signalA = signalAClass(input.N, input.LA);
		signalB = signalBClass(input.N, input.LB);

		tempHelper = {input.LB};
	}

	int init()
	{
		input.graph.dijkstra();
		HASHTABLE.resize(input.N);
		for (int i = 0; i <= input.N; ++i) 
        	HASHTABLE[i] = generate_random_uint32();  
		return 0;
	}

	int build_signalA()
	{
		auto &graph = input.graph;
		auto &dest_pos = input.dest_pos;

		int from = 0, to = 0;
		for (int i = 0; i < dest_pos.size(); ++i)
		{
			to = dest_pos[i];
			auto route = graph.routes[from][to];
			for (int j = 0; j < route.size() - 1; ++j)
			{
				int from2 = route[j], to2 = route[j + 1];
				graph.all_edge[from2][to2].dist -= 10;
				graph.all_edge[to2][from2].dist -= 10;
			}
			from = to;
		}
		graph.dijkstra();

		signalA.lists.clear();
		while(signalA.lists.size() < input.LA)
		{
			signalA.exists = 0;

			vector<int> tours;
			for (int i = 0; i < graph.V; ++i)
				tours.push_back(i);

			shuffleVector(tours);
			//shuffleVector(tours);
			
			while (1)
			{
				pair<int, int> swap;
				int best_gain = -INF;
				for (int i = 1; i < graph.V; ++i)
				{
					for (int j = i + 2; j < graph.V; ++j)
					{
						int gain = graph.sols[tours[i]][tours[(i + 1) % graph.V]] + graph.sols[tours[j]][tours[(j + 1) % graph.V]] - graph.sols[tours[i]][tours[j]] - graph.sols[tours[(i + 1) % graph.V]][tours[(j + 1) % graph.V]];
						if (best_gain < gain)
						{
							best_gain = gain;
							swap = {i, j};
						}
					}
				}
				if (best_gain > 0)
				{
					reverse(tours.begin() + swap.first + 1, tours.begin() + swap.second + 1);
				}
				else
				{
					break;
				}
			}

			
			for (int i = 0; i < tours.size() - 1; ++i)
			{
				int from = tours[i], to = tours[i + 1];
				auto route = graph.routes[from][to];
				for (int j = 0; j < route.size(); ++j)
				{
					if(signalA.lists.size() >= input.LA)	
						goto COMPLETE;

					if (signalA.lists.size() > 0 && signalA.lists.back() == route[j])
						continue;
					else if (signalA.exists[route[j]])
						continue;
					signalA.lists.push_back(route[j]);
					signalA.exists[route[j]] = 1;
				}
			}
		}
		
		COMPLETE:;

		signalA.lists.resize(input.LA);
		while (signalA.lists.size() < input.LA)
			signalA.lists.push_back(NONE);
		signalA.mapping();

		for(int i = 0; i < input.graph.all_edge.size(); ++i)
			for(int j = i + 1; j < input.graph.all_edge[i].size(); ++j)
				if(input.graph.all_edge[i][j].dist > 0)
					input.graph.all_edge[i][j].dist = input.graph.all_edge[j][i].dist = INITIAL_DIST;
		input.graph.dijkstra();
		
		cerr << "Complete Build SignalA" << endl;
		return 0;
	}

	int optimize()
	{
		logHandlerClass logHandler;
		timerClass timer;
		timer.start();

		int ret = 0;
		ret = build_signalA();

		logHandler.log("StartBeamSearch {}sec", timer.elapsed());
		logHandler.flush();

		// Prepare
		int turn;
		int beam_width = 30;
		const int max_turn = 3000;

		vector<stateClass> states, next_states;
		states.reserve(beam_width);

		stateClass root = {signalB};
		states.push_back(root);
		next_states.resize(beam_width, root);

		selectorClass selector = selectorClass(input.LB);

		nodes.resize(max_turn);

		// Start
		for (turn = 0; turn < max_turn; ++turn)
		{
			if(timer.elapsed() > 1.5) beam_width = 15;
			if (states.front().rank >= input.N)
				break;

			timer1.start();
			selector.clear();
			for (auto &state : states)
				state.expand(input, signalA, tempHelper, selector);
			e1 += timer1.elapsed();
			
			timer2.start();
			auto &candidates_all = selector.candidates_all;
			//candidates_all.resize(min(beam_width*100, (int) candidates_all.size()));
			sort(candidates_all.begin(), candidates_all.end(), greater<>());
			for(auto &candidate : candidates_all)
			{	
				selector.accept(signalA, states[candidate.parent].signal, input.graph, tempHelper, candidate);
				if(selector.candidates.size() >= beam_width) 
					break;
			}
			e2 += timer2.elapsed();
			
			auto &candidates = selector.candidates;
			const int size = min(beam_width, (int)candidates.size());
			//partial_sort(candidates.begin(), candidates.begin() + size, candidates.end(), greater<>());
			
#ifdef LOCALTEST
			logHandler.log("");
			logHandler.log("Turn [{}]", turn);
			logHandler.log("  └─Create Candidates {}", candidates.size());
			logHandler.log("  └─Create CandidatesALL {}", candidates_all.size());
#endif 
			nodes[turn] = candidates;
			for (int i = 0; i < size; i++)
			{
				auto &candidate = candidates[i];
				next_states[i] = states[candidate.parent];
#ifdef LOCALTEST
				if (i == 0)
				{
					auto [a, dist, n_achievements] = candidate.evaluator.decode();
					logHandler.log("   └─BestStateLog Rank {} Ache {} Dist {}", next_states[i].rank, n_achievements, dist);
				}					
#endif 
				next_states[i].transition(i, input, signalA, tempHelper, candidate);
			}
			states = next_states;
			
#ifdef LOCALTEST
			logHandler.flush();
			logHandler.clear();
#endif 
		}

		cerr << "Complete! " << timer.elapsed() << "sec" << endl;

		logHandler.log("TimeElapsed {} {} {}", e1, e2, e3);
		logHandler.flush();
		logHandler.clear();

		vector<candidateClass> growth_path;
		candidateClass best_node = nodes[--turn][0];
		growth_path.push_back(nodes[turn--][0]);
		int parent = best_node.parent;

		while (turn >= 0)
		{
			best_node = nodes[turn][parent];
			growth_path.push_back(nodes[turn--][parent]);
			parent = best_node.parent;
		}
		reverse(growth_path.begin(), growth_path.end());

		answer(root, growth_path);

		return ret;
	}

	int route_dfs(const int from, const int to, bitset<600> &visited, signalBClass &signal, vector<int> &route)
	{
		if (from == to)
			return 1;
		
		for (auto &edge : input.graph.edge_vec[from])
		{
			if (visited[edge->to] || !signal.exists[edge->to])
				continue;
			visited[edge->to] = 1;

			if (route_dfs(edge->to, to, visited, signal, route))
			{ 
				route.push_back(edge->to);
				return 1;
			}

			visited[edge->to] = 0;
		}
		return 0;
	}

	void answer(stateClass &root, vector<candidateClass> &growth_path)
	{

		signalBClass signalB = {input.N, input.LB};
		bitset<600> visited = 0;

		for (auto a : signalA.lists)
			cout << max(a, 0) << " ";
		cout << endl;

		int from = 0, to = 0, initial_state = 1;
		int n_achivement = 0;
		for (auto &node : growth_path)
		{
			auto &action = node.action;
			tempHelper.action = action;
			
			auto [left, right, index, direct, sig] = action.decode();
			if (!initial_state)
			{
				//to = action.sig;
				to = sig;
				vector<int> route;
				visited = 0; visited[signalA.lists[from]] = 1;
				route_dfs(signalA.lists[from], signalA.lists[to], visited, signalB, route);
				reverse(route.begin(), route.end());
				for (int i = 0; i < route.size(); ++i)
					cout << "m " << route[i] << endl;								
			}

			// if (action.direct == 1)
			// 	cout << "s " << (action.right - action.left) << " " << action.index << " " << action.left << endl;
			// else
			// 	cout << "s " << (action.right - action.left) << " " << action.index - action.right + action.left + 1 << " " << action.left << endl;

			if (direct == 1)
				cout << "s " << (right - left) << " " << index << " " << left << endl;
			else
				cout << "s " << (right - left) << " " << index - right + left + 1 << " " << left << endl;

			signalB.lists = tempHelper.new_signal(signalA, signalB);
			signalB.exists = tempHelper.exists;

			// from = action.sig;
			// to = action.index;
			//auto [left2, right2, index2, direct2, sig2] = 
			from = sig; 
			to = index;
			cout << "m " << signalA.lists[to] << endl;

			from = to;

			if(!initial_state)
			{
				int dist = INF;
				do
				{
					dist = INF;
					for(auto sig : signalB.lists)
					{						
						if(signalA.lists[sig] == input.dest_pos[n_achivement])
						{
							to = sig;
							vector<int> route;
							visited = 0; visited[signalA.lists[from]] = 1;
							route_dfs(signalA.lists[from], signalA.lists[to], visited, signalB, route);
							reverse(route.begin(), route.end());
							for (int i = 0; i < route.size(); i++)
								cout << "m " << route[i] << endl;
							
							++n_achivement;
							from = to;
							dist = 0;
						}	
					}
				} while (dist == 0);				
			}

			initial_state = 0;
		}
	}
};

int main()
{

	cerr << "Start AHC036" << endl;

	solverClass solver;
	int ret = 0;
	ret = solver.init();
	ret = solver.optimize();
	return ret;
}