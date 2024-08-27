#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stack>

using namespace std;

const int INF = 1e9;
const int NONE = -1;
const int INITIAL_DIST = 10000;

class UnionFinding
{
  public:

    UnionFinding(){}
    UnionFinding(int nsize)
    {
      par.reserve(nsize); 
      for(int i = 0; i < nsize; ++i)
      {
        par.push_back(i);
      }
    }

    vector<int> par;

    bool Union(int x, int y)
    {
      x = root(x);
      y = root(y);
      //y = par[y]; 
      par[y] = x;
      return true;
    }

    bool Find(int x, int y)
    {
      return root(x) == root(y);
    }

    int root(int x)
    {
      if(par[x] == x) return x;
      par[x] = root(par[x]);
      return par[x];
    }
};

  template<typename var>
inline var sq(var x)
{
  return x * x;
}

  template<typename var>
inline var cube(var x)
{
  return x * x * x;
}

static unsigned long xor128() 
{
  static unsigned long x=123456789, y=362436069, z=521288629, w=88675123;
  unsigned long t;
  t=(x^(x<<11)); x=y; y=z; z=w;
  return (w=(w^(w>>19))^(t^(t>>8)));
}

void myshuffle(vector<int> &ary,int size, bool flag)
{
  for(int i=0;i<size;i++)
  {
    int j = xor128()%size;
    //if(flag && (i == size - 1 || j == size - 1)) continue;
    if(flag && (i == 0 || j == 0)) continue;
    auto t = ary[i];
    ary[i] = ary[j];
    ary[j] = t;
  }
}

double rnd(){
	return (double) xor128() / ULONG_MAX;
}

class posClass;
class edgeClass;
class graphClass;

class posClass
{ 
  private:
    using data_type = int;
  public:
    posClass() = default;
    posClass(int id) : id(id) { }
    int id, done, pre;
    data_type cost;
};

class edgeClass
{
  private:
    using data_type = int;
  public:
    edgeClass() = default;
    edgeClass(string edge_name, int from, int to, data_type dist) : edge_name(edge_name), from(from), to(to), dist(dist)  { }
    string edge_name;
    int from, to;
    data_type dist;
    bool operator<(const edgeClass& other) const {
        return dist < other.dist;
    }
	bool operator>(const edgeClass& other) const {
		return dist > other.dist;
	}
};

template< typename T >
struct RadixHeap
{
  using uint = unsigned long long;
  vector< pair< uint, T > > v[33];
  uint size, last;

  RadixHeap() : size(0), last(0) {}

  bool empty() const { return size == 0; }
  inline int getbit(int a) { return a ? 32 - __builtin_clz(a) : 0; }
  void push(uint key, const T &value)
  { ++size; v[getbit(key ^ last)].emplace_back(key, value); }

  pair< uint, T > pop()
  {
    if(v[0].empty()) {
      int idx = 1;
      while(v[idx].empty()) ++idx;
      last = min_element(begin(v[idx]), end(v[idx]))->first;
      for(auto &p : v[idx]) v[getbit(p.first ^ last)].emplace_back(p);
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
	vector<vector<edgeClass*>> edge_vec;
    vector<posClass> pos_vec;
    vector<vector<int>> pre_routes, sols;
    vector<vector<vector<int>>> routes;

	void add_edge(int u, int v, data_type dist){
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

      for(auto &vec : pre_routes) vec.resize(V);
      for(auto &vec : sols) vec.resize(V, 0);
      for(auto &vec : routes) vec.resize(V);
    }

    int dijkstra(int from)
    {
      heap_type que;
      graphClass *g = this;

      const int n = g->V;
      for(auto &node : g->pos_vec)
      {
        node.done = false;
        node.cost = 1e8; 
        node.pre = -1;
      }

      g->pos_vec[from].cost = 0;

      que.push(0, from);

      while(!que.empty())
      {
        P p = que.pop();
        int v = p.second;
        auto *sNode = &g->pos_vec[v];
        if(sNode->cost < p.first) continue;

        auto &edges = g->edge_vec[v];
		for(int i = 0; i < edges.size(); ++i)
        {
          int to = edges[i]->to;
          data_type dist = edges[i]->dist;
		  auto *eNode = &g->pos_vec[to];
		  if(eNode->cost > sNode->cost + dist)
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

      for(int i = 0; i < n; ++i)
      {
        pre[i] = g->pos_vec[i].pre;
        sol[i] = g->pos_vec[i].cost;
   	  }
      for(int i = 0; i < n; ++i)
        rst[i] = getRoute(from, i);

      return 0;
    }

	int dijkstra(){
		init_dijkstra();
		for(int i = 0; i < V; ++i)
			dijkstra(i);	
		return 0;
	}

    int getdist(int from, int to){ return from == to ? 10000000: sols[from][to]; }

    vector<int> getRoute(int from , int to)
    {
      if(routes[from].size() > 0 && routes[from][to].size() > 0)
        return routes[from][to];
      vector<int> ret;
      ret.push_back(to);
      while(to != from && to >= 0)
      {
        to = pre_routes[from][to];
        if(0 <= to) ret.push_back(to);
      } 
      reverse(ret.begin(), ret.end());
      return ret;

    }
};

class signalAClass{
	private:
		int pos_size;
		int signal_size;
	public:
		signalAClass() = default;
		signalAClass(int pos_size, int signal_size) : pos_size(pos_size), signal_size(signal_size) { lists.resize(signal_size, NONE); }
		vector<int> lists;
		vector<vector<int>> indices;
		void mapping(){
			indices.clear();
			indices.resize(pos_size);
			for(int i = 0; i < lists.size(); ++i)
				if(lists[i] >= 0)
					indices[lists[i]].push_back(i);
			return;
		}
};
class signalBClass{
	private:
		int pos_size;
		int signal_size;
	public:
		signalBClass() = default;
		signalBClass(int pos_size, int signal_size) : pos_size(pos_size), signal_size(signal_size) { 
			lists.resize(signal_size, NONE); 
			exists.resize(pos_size, false);
		}
		vector<int> lists, exists;
};

class actionClass{
	private:
	public:
		actionClass() = default;
		int id, pos_id, range;
};

class stateClass{
	private:

	public:
		stateClass() = default;
		bool close = false;
		int id = 0, parent = -1, brother = -1, pos = 0;
		double score;
		actionClass action;

		bool operator<(const stateClass& other) const {
			return score < other.score;
		}
		bool operator>(const stateClass& other) const {
			return score > other.score;
		}
	
};

template<class T>
class ObjectPool {
    public:
        // 配列と同じようにアクセスできる
        T& operator[](int i) {
            return data_[i];
        }

        // 配列の長さを変更せずにメモリを確保する
        void reserve(size_t capacity) {
            data_.reserve(capacity);
        }

        // 要素を追加し、追加されたインデックスを返す
        int push(const T& x) {
            if (garbage_.empty()) {
                data_.push_back(x);
                return data_.size() - 1;
            } else {
                int i = garbage_.top();
                garbage_.pop();
                data_[i] = x;
                return i;
            }
        }

        // 要素を（見かけ上）削除する
        void pop(int i) {
            garbage_.push(i);
        }

        // 使用した最大のインデックス(+1)を得る
        // この値より少し大きい値をreserveすることでメモリの再割り当てがなくなる
        size_t size() {
            return data_.size();
        }

    private:
        vector<T> data_;
        stack<int> garbage_;
};

// 連想配列
// Keyにハッシュ関数を適用しない
// open addressing with linear probing
// unordered_mapよりも速い
// nは格納する要素数よりも4~16倍ほど大きくする
template <class Key, class T>
struct HashMap {
    public:
        explicit HashMap(uint32_t n) {
            n_ = n;
            valid_.resize(n_, false);
            data_.resize(n_);
        }

        // 戻り値
        // - 存在するならtrue、存在しないならfalse
        // - index
        pair<bool,int> get_index(Key key) const {
            Key i = key % n_;
            while (valid_[i]) {
                if (data_[i].first == key) {
                    return {true, i};
                }
                if (++i == n_) {
                    i = 0;
                }
            }
            return {false, i};
        }

        // 指定したindexにkeyとvalueを格納する
        void set(int i, Key key, T value) {
            valid_[i] = true;
            data_[i] = {key, value};
        }

        // 指定したindexのvalueを返す
        T get(int i) const {
            assert(valid_[i]);
            return data_[i].second;
        }

        void clear() {
            fill(valid_.begin(), valid_.end(), false);
        }

    private:
        uint32_t n_;
        vector<bool> valid_;
        vector<pair<Key,T>> data_;
};

using Hash = uint32_t; // TODO

// 状態遷移を行うために必要な情報
// メモリ使用量をできるだけ小さくしてください
struct Action {
    // TODO

    Action() {
        // TODO
    }
};

using Cost = int; // TODO

// 状態のコストを評価するための構造体
// メモリ使用量をできるだけ小さくしてください
struct Evaluator {
    // TODO

    Evaluator() {
        // TODO
    }

    // 低いほどよい
    Cost evaluate() const {
        // TODO
    }
};

// 展開するノードの候補を表す構造体
struct Candidate {
    Action action;
    Evaluator evaluator;
    Hash hash;
    int parent;
    Cost cost;

    Candidate(Action action, Evaluator evaluator, Hash hash, int parent, Cost cost) :
        action(action),
        evaluator(evaluator),
        hash(hash),
        parent(parent),
        cost(cost) {}
};

// ビームサーチの設定
struct Config {
    int max_turn;
    size_t beam_width;
    size_t nodes_capacity;
    uint32_t hash_map_capacity;
};

// 削除可能な優先度付きキュー
using MaxSegtree = atcoder::segtree<
    pair<Cost,int>,
    [](pair<Cost,int> a, pair<Cost,int> b){
        if (a.first >= b.first) {
            return a;
        } else {
            return b;
        }
    },
    []() { return make_pair(numeric_limits<Cost>::min(), -1); }
>;

// ノードの候補から実際に追加するものを選ぶクラス
// ビーム幅の個数だけ、評価がよいものを選ぶ
// ハッシュ値が一致したものについては、評価がよいほうのみを残す
class Selector {
    public:
        explicit Selector(const Config& config) :
            hash_to_index_(config.hash_map_capacity)
        {
            beam_width = config.beam_width;
            candidates_.reserve(beam_width);
            full_ = false;
            st_original_.resize(beam_width);
        }

        // 候補を追加する
        // ターン数最小化型の問題で、candidateによって実行可能解が得られる場合にのみ finished = true とする
        // ビーム幅分の候補をCandidateを追加したときにsegment treeを構築する
        
		/*void push(Action action, const Evaluator& evaluator, Hash hash, int parent, bool finished) {
            Cost cost = evaluator.evaluate();
            if (finished) {
                finished_candidates_.emplace_back(Candidate(action, evaluator, hash, parent, cost));
                return;
            }
            if (full_ && cost >= st_.all_prod().first) {
                // 保持しているどの候補よりもコストが小さくないとき
                return;
            }
            auto [valid, i] = hash_to_index_.get_index(hash);

            if (valid) {
                int j = hash_to_index_.get(i);
                if (hash == candidates_[j].hash) {
                    // ハッシュ値が等しいものが存在しているとき
                    if (cost < candidates_[j].cost) {
                        // 更新する場合
                        candidates_[j] = Candidate(action, evaluator, hash, parent, cost);
                        if (full_) {
                            st_.set(j, {cost, j});
                        }
                    }
                    return;
                }
            }
            if (full_) {
                // segment treeが構築されている場合
                int j = st_.all_prod().second;
                hash_to_index_.set(i, hash, j);
                candidates_[j] = Candidate(action, evaluator, hash, parent, cost);
                st_.set(j, {cost, j});
            } else {
                // segment treeが構築されていない場合
                hash_to_index_.set(i, hash, candidates_.size());
                candidates_.emplace_back(Candidate(action, evaluator, hash, parent, cost));

                if (candidates_.size() == beam_width) {
                    // 保持している候補がビーム幅分になったとき
                    construct_segment_tree();
                }
            }
        }*/

        // 選んだ候補を返す
        const vector<Candidate>& select() const {
            return candidates_;
        }

        // 実行可能解が見つかったか
        bool have_finished() const {
            return !finished_candidates_.empty();
        }

        // 実行可能解に到達する「候補」を返す
        vector<Candidate> get_finished_candidates() const {
            return finished_candidates_;
        }

        void clear() {
            candidates_.clear();
            hash_to_index_.clear();
            full_ = false;
        }

    private:
        size_t beam_width;
        vector<Candidate> candidates_;
        HashMap<Hash,int> hash_to_index_;
        bool full_;
        vector<pair<Cost,int>> st_original_;
        MaxSegtree st_;
        vector<Candidate> finished_candidates_;

        void construct_segment_tree() {
            full_ = true;
            for (size_t i = 0; i < beam_width; ++i) {
                st_original_[i] = {candidates_[i].cost, i};
            }
            st_ = MaxSegtree(st_original_);
        }
};

// 深さ優先探索に沿って更新する情報をまとめたクラス
class State {
    public:
        explicit State(/* const Input& input */) {
            // TODO
        }

        // 次の状態候補を全てselectorに追加する
        // 引数
        //   evaluator : 今の評価器
        //   hash      : 今のハッシュ値
        //   parent    : 今のノードID（次のノードにとって親となる）
        void expand(const Evaluator& evaluator, Hash hash, int parent, Selector& selector) {
            // TODO
        }

        // actionを実行して次の状態に遷移する
        void move_forward(Action action) {
            // TODO
        }

        // actionを実行する前の状態に遷移する
        // 今の状態は、親からactionを実行して遷移した状態である
        void move_backward(Action action) {
            // TODO
        }

    private:
        // TODO
};

// 探索木（二重連鎖木）のノード
struct Node {
    Action action;
    Evaluator evaluator;
    Hash hash;
    int parent, child, left, right;

    // 根のコンストラクタ
    Node(Action action, const Evaluator& evaluator, Hash hash) :
        action(action),
        evaluator(evaluator),
        hash(hash),
        parent(-1),
        child(-1),
        left(-1),
        right(-1) {}

    // 通常のコンストラクタ
    Node(const Candidate& candidate, int right) :
        action(candidate.action),
        evaluator(candidate.evaluator),
        hash(candidate.hash),
        parent(candidate.parent),
        child(-1),
        left(-1),
        right(right) {}
};

// 二重連鎖木に対する操作をまとめたクラス
class Tree {
    public:
        explicit Tree(const State& state, size_t nodes_capacity, const Node& root) :
            state_(state)
        {
            nodes_.reserve(nodes_capacity);
            root_ = nodes_.push(root);
        }

        // 状態を更新しながら深さ優先探索を行い、次のノードの候補を全てselectorに追加する
        void dfs(Selector& selector) {
            update_root();

            int v = root_;
            while (true) {
                v = move_to_leaf(v);
                state_.expand(nodes_[v].evaluator, nodes_[v].hash, v, selector);
                v = move_to_ancestor(v);
                if (v == root_) {
                    break;
                }
                v = move_to_right(v);
            }
        }

        // 根からノードvまでのパスを取得する
        vector<Action> get_path(int v) {
            // cerr << nodes_.size() << endl;

            vector<Action> path;
            while (nodes_[v].parent != -1) {
                path.push_back(nodes_[v].action);
                v = nodes_[v].parent;
            }
            reverse(path.begin(), path.end());
            return path;
        }

        // 新しいノードを追加する
        int add_leaf(const Candidate& candidate) {
            int parent = candidate.parent;
            int sibling = nodes_[parent].child;
            int v = nodes_.push(Node(candidate, sibling));

            nodes_[parent].child = v;

            if (sibling != -1) {
                nodes_[sibling].left = v;
            }
            return v;
        }

        // ノードvに子がいなかった場合、vと不要な先祖を削除する
        void remove_if_leaf(int v) {
            if (nodes_[v].child == -1) {
                remove_leaf(v);
            }
        }

        // 最も評価がよいノードを返す
        int get_best_leaf(const vector<int>& last_nodes) {
            assert(!last_nodes.empty());
            int ret = last_nodes[0];
            for (int v : last_nodes) {
                if (nodes_[v].evaluator.evaluate() < nodes_[ret].evaluator.evaluate()) {
                    ret = v;
                }
            }
            return ret;
        }

    private:
        State state_;
        ObjectPool<Node> nodes_;
        int root_;

        // 根から一本道の部分は往復しないようにする
        void update_root() {
            int child = nodes_[root_].child;
            while (child != -1 && nodes_[child].right == -1) {
                root_ = child;
                state_.move_forward(nodes_[child].action);
                child = nodes_[child].child;
            }
        }

        // ノードvの子孫で、最も左にある葉に移動する
        int move_to_leaf(int v) {
            int child = nodes_[v].child;
            while (child != -1) {
                v = child;
                state_.move_forward(nodes_[child].action);
                child = nodes_[child].child;
            }
            return v;
        }

        // ノードvの先祖で、右への分岐があるところまで移動する
        int move_to_ancestor(int v) {
            while (v != root_ && nodes_[v].right == -1) {
                state_.move_backward(nodes_[v].action);
                v = nodes_[v].parent;
            }
            return v;
        }

        // ノードvの右のノードに移動する
        int move_to_right(int v) {
            state_.move_backward(nodes_[v].action);
            v = nodes_[v].right;
            state_.move_forward(nodes_[v].action);
            return v;
        }

        // 不要になった葉を再帰的に削除する
        void remove_leaf(int v) {
            while (true) {
                int left = nodes_[v].left;
                int right = nodes_[v].right;
                if (left == -1) {
                    int parent = nodes_[v].parent;

                    if (parent == -1) {
                        cerr << "ERROR: root is removed" << endl;
                        exit(-1);
                    }
                    nodes_.pop(v);
                    nodes_[parent].child = right;
                    if (right != -1) {
                        nodes_[right].left = -1;
                        return;
                    }
                    v = parent;
                } else {
                    nodes_.pop(v);
                    nodes_[left].right = right;
                    if (right != -1) {
                        nodes_[right].left = left;
                    }
                    return;
                }
            }
        }
};

// ビームサーチを行う関数
vector<Action> beam_search(const Config& config, State state, Node root) {
    Tree tree(state, config.nodes_capacity, root);

    // 探索中のノード集合
    vector<int> curr_nodes;
    curr_nodes.reserve(config.beam_width);
    // 本来は curr_nodes = {state.root_} とすべきだが, 省略しても問題ない

    // 新しいノードの集合
    vector<int> next_nodes;
    next_nodes.reserve(config.beam_width);

    // 新しいノード候補の集合
    Selector selector(config);

    for (int turn = 0; turn < config.max_turn; ++turn) {
        // Euler Tour で selector に候補を追加する
        tree.dfs(selector);

        if (selector.have_finished()) {
            // ターン数最小化型の問題で実行可能解が見つかったとき
            Candidate candidate = selector.get_finished_candidates()[0];
            vector<Action> ret = tree.get_path(candidate.parent);
            ret.push_back(candidate.action);
            return ret;
        }
        // 新しいノードを追加する
        for (const Candidate& candidate : selector.select()) {
            next_nodes.push_back(tree.add_leaf(candidate));
        }
        if (next_nodes.empty()) {
            // 新しいノードがないとき
            cerr << "ERROR: Failed to find any valid solution" << endl;
            return {};
        }
        // 不要なノードを再帰的に削除する
        for (int v : curr_nodes) {
            tree.remove_if_leaf(v);
        }
        // ダブルバッファリングで配列を使い回す
        swap(curr_nodes, next_nodes);
        next_nodes.clear();

        selector.clear();
    }
    // ターン数固定型の問題で全ターンが終了したとき
    int best_leaf = tree.get_best_leaf(curr_nodes);
    return tree.get_path(best_leaf);
}


class solverClass{
	private:
		int N, M, T, LA, LB;
		signalAClass signalA;
		signalBClass signalB;
		vector<int> dest_pos;
		vector<posClass> pos;
		graphClass graph;
	public:
		solverClass(){
			input(); 
		}
		void input(){
			cin >> N >> M >> T >> LA >> LB;
			
			for(int i = 0; i < N; ++i) pos.push_back(posClass(i));
			graph = graphClass(N, pos);
			
			for (int m = 0; m < M; ++m){
				int u, v; cin >> u >> v;
				graph.add_edge(u, v, INITIAL_DIST);	
			}
			for(int t = 0; t < T; ++t){
				int ti; cin >> ti; 
				dest_pos.push_back(ti);				
			}

			signalA = signalAClass(N, LA);
			signalB = signalBClass(N, LB);
			
			return ;
		}

		int init(){
			graph.dijkstra();
			return 0;
		}

		int build_signalA(){
			int best_dist = 1e9, dist = 0;
			vector<int> tours;
			for(int i = 0; i < graph.V; ++i) tours.push_back(i);
			myshuffle(tours, graph.V, true);
			//myshuffle(tours, graph.V, true);

			int from = 0, to = 0;
			for(int i = 0; i < dest_pos.size(); ++i){
				to = dest_pos[i];
				auto route = graph.routes[from][to];
				for(int j = 0; j < route.size() - 1; ++j){
					int from2 = route[j], to2 = route[j + 1];
					graph.all_edge[from2][to2].dist -= 10;
					graph.all_edge[to2][from2].dist -= 10;
				}
				from = to;
			}
			graph.dijkstra();
			for(int i = 0; i < graph.V; ++i) 
				dist += graph.sols[tours[i]][tours[(i+1)%graph.V]];

			while(1){
				pair<int, int> swap;
				int best_gain = -INF;
				for(int i = 1; i < graph.V; ++i){
					for(int j = i + 2; j < graph.V; ++j){
						int	gain = graph.sols[tours[i]][tours[(i+1)%graph.V]] + graph.sols[tours[j]][tours[(j+1)%graph.V]]
								- graph.sols[tours[i]][tours[j]] - graph.sols[tours[(i+1)%graph.V]][tours[(j+1)%graph.V]];
						if(best_gain < gain){
							best_gain = gain;
							swap = {i, j};
						}
					}
				}	
				if(best_gain > 0){
					reverse(tours.begin() + swap.first + 1, tours.begin() + swap.second + 1);
					dist -= best_gain;
				}
				else {
					break;
				}
			}
			for(int i = 0; i < signalA.lists.size(); ++i)
				signalA.lists[i] = NONE;
			for(int i = 0; i < graph.V; ++i)
				signalA.lists[i] = i;
			for(int i = 0; i < signalA.lists.size(); ++i)
				cout << max(0, signalA.lists[i]) << " ";
			cout << endl;

			for(int i = 0; i < tours.size() - 1; ++i){
				int from = tours[i], to = tours[i+1];
				auto route = graph.routes[from][to];
				for(int j = 1; j < route.size(); ++j){
					cout << "s 1 " << route[j] << " 0" << endl; 
					cout << "m " << route[j] << endl;	
				}
			}	

			signalA.mapping();

			return 0;
		}

		int optimize(){

			int ret = 0;
			ret = build_signalA();	

			ret = 1;
			while(ret){

				ret = 0;

						


			}


			return 0;
		}
};

int main(){
	solverClass solver;
	int ret = 0; 
	ret = solver.init();
	ret = solver.optimize();
	return 0;
}