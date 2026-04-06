import { Routes, Route, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import ProductList from './pages/ProductList';
import ProductDetail from './pages/ProductDetail';
import { X, Trash2 } from 'lucide-react';

function App() {
  const [isCartOpen, setIsCartOpen] = useState(false);
  const [cartItems, setCartItems] = useState([]);
  const [toast, setToast] = useState(null);
  const location = useLocation();

  // Scroll to top on route change
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  const addToCart = (product, size, qty) => {
    setCartItems(prev => {
      const existing = prev.find(item => item.id === product.id && item.size === size);
      if (existing) {
        return prev.map(item => item.id === product.id && item.size === size 
          ? { ...item, qty: item.qty + qty } : item);
      }
      return [...prev, { ...product, size, qty }];
    });
    
    // Show toast
    setToast(`已將 ${product.title} 加入購物車`);
    setTimeout(() => setToast(null), 3000);
    setIsCartOpen(true);
  };

  const removeFromCart = (id, size) => {
    setCartItems(prev => prev.filter(item => !(item.id === id && item.size === size)));
  };

  const cartTotal = cartItems.reduce((acc, item) => acc + (item.price * item.qty), 0);
  const cartCount = cartItems.reduce((acc, item) => acc + item.qty, 0);

  return (
    <>
      <Header cartCount={cartCount} openCart={() => setIsCartOpen(true)} />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/shop" element={<ProductList />} />
          <Route path="/product/:id" element={<ProductDetail addToCart={addToCart} />} />
        </Routes>
      </main>
      <Footer />

      {/* Cart Drawer */}
      <div className={`overlay ${isCartOpen ? 'active' : ''}`} onClick={() => setIsCartOpen(false)}></div>
      <div className={`drawer ${isCartOpen ? 'active' : ''}`}>
        <div className="drawer-header">
          <h3>購物車 ({cartCount})</h3>
          <button onClick={() => setIsCartOpen(false)}><X size={24} /></button>
        </div>
        <div className="drawer-body">
          {cartItems.length === 0 ? (
            <div style={{ textAlign: 'center', marginTop: '3rem', color: 'var(--color-text-light)' }}>
              購物車是空的
            </div>
          ) : (
            cartItems.map((item, idx) => (
              <div key={idx} className="cart-item">
                <img src={item.images[0]} alt={item.title} className="cart-img" />
                <div className="cart-details">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <h4 className="cart-title">{item.title}</h4>
                    <button onClick={() => removeFromCart(item.id, item.size)} style={{ color: 'var(--color-text-light)' }}><Trash2 size={16} /></button>
                  </div>
                  <div className="cart-meta">Size: {item.size}</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px' }}>
                    <div className="qty-control">
                      <button onClick={() => {
                        if (item.qty > 1) setCartItems(prev => prev.map((x, i) => i === idx ? {...x, qty: x.qty - 1} : x));
                      }}>-</button>
                      <span>{item.qty}</span>
                      <button onClick={() => {
                        setCartItems(prev => prev.map((x, i) => i === idx ? {...x, qty: x.qty + 1} : x));
                      }}>+</button>
                    </div>
                    <div className="cart-price">NT$ {(item.price * item.qty).toLocaleString()}</div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
        {cartItems.length > 0 && (
          <div className="drawer-footer">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: 'bold' }}>
              <span>總計金額</span>
              <span>NT$ {cartTotal.toLocaleString()}</span>
            </div>
            <button className="btn btn-primary" style={{ width: '100%' }}>進入結帳</button>
          </div>
        )}
      </div>

      {/* Toast */}
      {toast && (
        <div className="toast-container">
          <div className="toast">{toast}</div>
        </div>
      )}
    </>
  );
}

export default App;
