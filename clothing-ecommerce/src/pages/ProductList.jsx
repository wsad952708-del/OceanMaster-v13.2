import { useState } from 'react';
import { Link } from 'react-router-dom';
import { products } from '../data';
import { ChevronDown, SlidersHorizontal } from 'lucide-react';

const ProductCard = ({ product }) => (
  <div className="product-card">
    {product.isNew && <div className="badge new">NEW</div>}
    {product.isSale && !product.isNew && <div className="badge sale">SALE</div>}
    <div className="product-image-wrap">
      <Link to={`/product/${product.id}`}>
        <img src={product.images[0]} alt={product.title} className="product-image primary-img" />
        {product.images[1] && <img src={product.images[1]} alt={product.title} className="product-image hover-img" />}
      </Link>
      <Link to={`/product/${product.id}`} className="quick-add">查看商品細節</Link>
    </div>
    <div className="product-info">
      <div className="product-category">{product.category === 'baby' ? '嬰幼兒 (0-2歲)' : product.category === 'toddler' ? '小童 (3-6歲)' : '大童 (7-12歲)'}</div>
      <h3 className="product-title"><Link to={`/product/${product.id}`}>{product.title}</Link></h3>
      <div className="product-price">
        {product.originalPrice && <span style={{ textDecoration: 'line-through', color: '#aaa', marginRight: '8px', fontSize: '0.9rem' }}>NT$ {product.originalPrice}</span>}
        NT$ {product.price}
      </div>
    </div>
  </div>
);

const ProductList = () => {
  const [filter, setFilter] = useState('All');

  const filteredProducts = filter === 'All' 
    ? products 
    : products.filter(p => p.category === filter);

  return (
    <div>
      <div className="shop-banner">
        <div className="container-sm">
          <span className="section-subtitle">Online Store</span>
          <h1 style={{ fontSize: '3rem', margin: '1rem 0', fontFamily: 'var(--font-serif)', color: 'var(--color-primary-dark)' }}>全部童裝系列</h1>
          <p style={{ color: 'var(--color-text-light)', fontSize: '1.1rem' }}>為每一個成長階段挑選最合適的純淨材質，由內而外散發孩子最純真的氣質。</p>
        </div>
      </div>

      <div className="container shop-layout">
        <aside className="sidebar">
          <div className="filter-group">
            <h3 className="filter-title">商品分類</h3>
            <ul className="filter-list">
              <li>
                <label onClick={() => setFilter('All')} style={{ color: filter === 'All' ? 'var(--color-primary)' : '', fontWeight: filter === 'All' ? '600' : '400' }}>
                  全部商品 ({products.length})
                </label>
              </li>
              <li>
                <label onClick={() => setFilter('baby')} style={{ color: filter === 'baby' ? 'var(--color-primary)' : '', fontWeight: filter === 'baby' ? '600' : '400' }}>
                  嬰幼兒 (0-2歲) ({products.filter(p=>p.category==='baby').length})
                </label>
              </li>
              <li>
                <label onClick={() => setFilter('toddler')} style={{ color: filter === 'toddler' ? 'var(--color-primary)' : '', fontWeight: filter === 'toddler' ? '600' : '400' }}>
                  小童 (3-6歲) ({products.filter(p=>p.category==='toddler').length})
                </label>
              </li>
              <li>
                <label onClick={() => setFilter('kids')} style={{ color: filter === 'kids' ? 'var(--color-primary)' : '', fontWeight: filter === 'kids' ? '600' : '400' }}>
                  大童 (7-12歲) ({products.filter(p=>p.category==='kids').length})
                </label>
              </li>
            </ul>
          </div>

          <div className="filter-group">
            <h3 className="filter-title">尺寸 Size</h3>
            <ul className="filter-list">
              <li><label><input type="checkbox" /> 60-80cm (嬰幼兒)</label></li>
              <li><label><input type="checkbox" /> 90-110cm (小童)</label></li>
              <li><label><input type="checkbox" /> 120-150cm (大童)</label></li>
            </ul>
          </div>

          <div className="filter-group">
            <h3 className="filter-title">庫存狀態</h3>
            <ul className="filter-list">
              <li><label><input type="checkbox" /> 現貨供應 In Stock</label></li>
              <li><label><input type="checkbox" /> 預購中 Pre-order</label></li>
            </ul>
          </div>
        </aside>

        <div className="shop-content">
          <div className="shop-toolbar">
            <div style={{ color: 'var(--color-text-light)', fontSize: '0.95rem' }}>
              顯示 <strong>{filteredProducts.length}</strong> 件商品
            </div>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <select className="sort-select">
                <option value="featured">推薦排序</option>
                <option value="newest">最新上架</option>
                <option value="price-low">價格：由低至高</option>
                <option value="price-high">價格：由高至低</option>
              </select>
            </div>
          </div>

          <div className="grid grid-3">
            {filteredProducts.map(product => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
          
          <div className="text-center" style={{ marginTop: '4rem' }}>
            <button className="btn btn-outline" style={{ padding: '0.8rem 3rem' }}>載入更多商品</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductList;
