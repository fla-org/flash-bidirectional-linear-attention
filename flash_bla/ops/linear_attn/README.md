Define $\phi(\cdot)$ as a feature map, such as $\text{elu}(\cdot) + 1$, and let $N$ be the sequence length. Linear attention can be written as:

$$
\boldsymbol{o}_i = \frac{\phi(\boldsymbol{q}_i) \sum _{j=1}^N \left( \phi(\boldsymbol{k}_j)^\top \boldsymbol{v}_j \right)}{\phi(\boldsymbol{q}_i) \sum _{j=1}^N \phi(\boldsymbol{k}_j)}
$$

Let $\boldsymbol{z} = \sum _{j=1}^N \phi(\boldsymbol{k}_j)$. In practical code implementation, to avoid potential overflow caused by summation, $\phi(\boldsymbol{k})$ should be scaled by $N^{-1}$:

$$
\boldsymbol{o}_i = \frac{\phi(\boldsymbol{q}_i) \sum _{j=1}^N \left( N^{-1} \cdot \phi(\boldsymbol{k}_j)^\top \boldsymbol{v}_j \right)}{\phi(\boldsymbol{q}_i) \sum _{j=1}^N \left( N^{-1} \cdot \phi(\boldsymbol{k}_j) \right)}
$$

In this case, $\boldsymbol{z}$ can be interpreted as the sum of $\phi(\boldsymbol{k})$ replaced with the mean of $\phi(\boldsymbol{k})$:

```python
scale = k.shape[-2] ** -1.0
        
z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)
s = (k.transpose(-2, -1) * scale) @ v
o = q @ s / (z + 1e-6)
```
