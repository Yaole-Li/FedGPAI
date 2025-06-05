### 算法描述
**Algorithm 1 FedRep**

**Parameters:** Participation rate \( r \), step sizes \( \alpha, \eta \); number of local updates \( \tau \); number of communication rounds \( T \).

Initialize \( \phi^0, h_1^0, \ldots, h_n^0 \)

**for** \( t = 1, 2, \ldots, T \) **do**

- Server receives a batch of clients \( \mathcal{I}^t \) of size \( rn \)
- Server sends current representation \( \phi^t \) to these clients

&nbsp;&nbsp;&nbsp;&nbsp;**for each** client \( i \in \mathcal{I}^t \) **do**

- Client \( i \) initializes \( h_i^t \leftarrow h_i^{t-1,\tau} \)
- Client \( i \) makes \( \tau \) updates to its head \( h_i^t \):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** \( s = 1 \) **to** \( \tau \) **do**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\[
h_i^{t,s} \leftarrow \text{GRD}(f_i(h_i^{t,s}, \phi^t), h_i^{t,s}, \alpha)
\]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end for**

- Client \( i \) locally updates the representation as:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\[
\phi_i^{t+1} \leftarrow \text{GRD}(f_i(h_i^{t,\tau}, \phi^t), \phi^t, \alpha)
\]

- Client \( i \) sends updated representation \( \phi_i^{t+1} \) to server

&nbsp;&nbsp;&nbsp;&nbsp;**end for**

&nbsp;&nbsp;&nbsp;&nbsp;**for each** client \( i \notin \mathcal{I}^t \), **do**

- Set \( h_i^{t,\tau} \leftarrow h_i^{t-1,\tau} \)

&nbsp;&nbsp;&nbsp;&nbsp;**end for**

- Server computes the new representation as:

\[
\phi^{t+1} = \frac{1}{rn} \sum_{i \in \mathcal{I}^t} \phi_i^{t+1}
\]

**end for**


### 特殊版本
---

**Algorithm 2 FedRep for linear regression**

**Input:** Step size \( \eta \); number of rounds \( T \); participation rate \( r \).

**Initialization:** Each client \( i \in [n] \) sends \( \mathbf{Z}_i := \frac{1}{m} \sum_{j=1}^{m} (y_i^{0,j})^2 \mathbf{x}_i^{0,j} (\mathbf{x}_i^{0,j})^\top \) to server, server computes  
\[
\mathbf{U} \mathbf{D} \mathbf{U}^\top \leftarrow \text{rank-}k \text{ SVD}\left( \frac{1}{n} \sum_{i=1}^{n} \mathbf{Z}_i \right)
\]  
Server initializes \( \mathbf{B}^0 \leftarrow \mathbf{U} \)

**for** \( t = 1, 2, \ldots, T \) **do**

- Server receives a subset \( \mathcal{I}^t \) of clients of size \( rn \)
- Server sends current representation \( \mathbf{B}^t \) to these clients

&nbsp;&nbsp;&nbsp;&nbsp;**for** \( i \in \mathcal{I}^t \) **do**

**Client update:**

- Client \( i \) samples a fresh batch of \( m \) samples
- Client \( i \) updates \( \mathbf{w}_i \):

\[
\mathbf{w}_i^{t+1} \leftarrow \arg\min_{\mathbf{w}} \hat{f}_i^t(\mathbf{w}, \mathbf{B}^t)
\]

- Client \( i \) updates representation:

\[
\mathbf{B}_i^{t+1} \leftarrow \mathbf{B}^t - \eta \nabla_{\mathbf{B}} \hat{f}_i^t(\mathbf{w}_i^{t+1}, \mathbf{B}^t)
\]

- Client \( i \) sends \( \mathbf{B}_i^{t+1} \) to the server

&nbsp;&nbsp;&nbsp;&nbsp;**end for**

- **Server update:**  
\[
\mathbf{B}^{t+1} \leftarrow \frac{1}{rn} \sum_{i \in \mathcal{I}^t} \mathbf{B}_i^{t+1}
\]

**end for**
