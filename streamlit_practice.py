import streamlit as st

st.markdown(
    """
    ```
    # 개별 데이터 생존 확률 계산
    data = [pclass, sex, age, sex*pclass, cabin_class]
    log_odds = sum(data * model.coef_[0]) + model.intercept_[0]
    odds = np.exp(log_odds)
    p_death = round(((1/(1+odds)) * 100), 2)
    p_surv = round((100 - p_death), 2)
    ```"""
    )

def render_function(fn):
    st.markdown(
        """
```python
%s

```"""
        % inspect.getsource(fn)
    )