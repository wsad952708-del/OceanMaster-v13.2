import { Link } from 'react-router-dom';
import { products } from '../data';
import { ArrowRight, Leaf, ShieldCheck, Heart, Sparkles } from 'lucide-react';
import { useGlobalReveal } from '../hooks/useScrollReveal';

const ProductCard = ({ product }) => {
  return (
  <div className="product-card reveal-on-scroll">
    {product.isNew && <div className="badge new">NEW</div>}
    {product.isSale && !product.isNew && <div className="badge sale">SALE</div>}
    <div className="product-image-wrap">
      <Link to={`/product/${product.id}`}>
        <img src={product.images[0]} alt={product.title} className="product-image primary-img" />
        {product.images[1] && <img src={product.images[1]} alt={product.title} className="product-image hover-img" />}
      </Link>
      <div className="card-quick-add-tray">
        <div className="tray-title">快速加入購物車</div>
        <div className="tray-sizes">
          {product.sizes.map(size => (
            <button key={size} className="tray-size-btn">{size}</button>
          ))}
        </div>
      </div>
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
)};

const Home = () => {
  useGlobalReveal();
  const newArrivals = products.filter(p => p.isNew).slice(0, 4);

  return (
    <div>
      {/* 1. Hero Parallax / Ken Burns */}
      <section className="hero">
        <div className="hero-bg" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1473968512647-3e447244af8f?auto=format&fit=crop&w=2000&q=80)' }}></div>
        <div className="hero-content">
          <span className="hero-subtitle">Spring / Summer 2026</span>
          <h1 style={{ fontFamily: 'var(--font-serif)' }}>陪伴寶貝每一個純真時刻</h1>
          <p>天然有機棉、舒適無毒染料，給孩子最溫柔的親膚呵護。</p>
          <Link to="/shop" className="btn btn-primary" style={{ marginTop: '1rem', backgroundColor: 'var(--color-accent)', border: 'none' }}>
            選購當季新品 <ArrowRight size={18} style={{ marginLeft: '10px' }} />
          </Link>
        </div>
      </section>

      {/* Ticker / Marquee */}
      <div className="ticker-wrap">
        <div className="ticker">
          <div className="ticker-item"><Sparkles size={14}/> 100% ORGANIC COTTON</div>
          <div className="ticker-item"><Sparkles size={14}/> FREE SHIPPING OVER NT$1500</div>
          <div className="ticker-item"><Sparkles size={14}/> ECO-FRIENDLY MATERIALS</div>
          <div className="ticker-item"><Sparkles size={14}/> OEKO-TEX® CERTIFIED</div>
          <div className="ticker-item"><Sparkles size={14}/> 100% ORGANIC COTTON</div>
          <div className="ticker-item"><Sparkles size={14}/> FREE SHIPPING OVER NT$1500</div>
          <div className="ticker-item"><Sparkles size={14}/> ECO-FRIENDLY MATERIALS</div>
          <div className="ticker-item"><Sparkles size={14}/> OEKO-TEX® CERTIFIED</div>
        </div>
      </div>

      {/* 2. Features (Why Choose Us) */}
      <section className="features-section" style={{ position: 'relative', zIndex: 1, background: 'rgba(255,255,255,0.4)', backdropFilter: 'blur(10px)' }}>
        <div className="container grid grid-3">
          <div className="feature-box reveal-on-scroll">
            <Leaf size={40} className="feature-icon" strokeWidth={1.5} />
            <h4>100% 天然純棉</h4>
            <p>嚴選 GOTS 認證有機棉，手感輕柔、透氣排汗，守護寶寶敏感肌膚。</p>
          </div>
          <div className="feature-box reveal-on-scroll">
            <ShieldCheck size={40} className="feature-icon" strokeWidth={1.5} />
            <h4>歐盟安全認證</h4>
            <p>零甲醛、無螢光劑，從染料到縫線皆符合國際最高嬰幼兒安全標準。</p>
          </div>
          <div className="feature-box reveal-on-scroll">
            <Heart size={40} className="feature-icon" strokeWidth={1.5} />
            <h4>舒適活動剪裁</h4>
            <p>專為東方孩童體型開發，隱藏式縫線與無感標籤，跑跳零負擔。</p>
          </div>
        </div>
      </section>

      {/* 3. Shop by Category (Banner Grid) */}
      <section style={{ padding: '5rem 0' }}>
        <div className="container reveal-on-scroll">
          <span className="section-subtitle">Categories</span>
          <h2 className="section-title"><span>探索</span>系列童裝</h2>
          
          <div className="banner-grid reveal-on-scroll">
            <Link to="/shop?category=baby" className="cat-banner">
              <img src="https://images.unsplash.com/photo-1519689680058-324335c77eba?auto=format&fit=crop&w=1200&q=80" alt="Baby" />
              <div className="cat-content">
                <h3>嬰幼兒 (0-2歲)</h3>
                <span>Shop Baby</span>
              </div>
            </Link>
            <Link to="/shop?category=accessories" className="cat-banner tall">
              <img src="https://images.unsplash.com/photo-1522771730848-fa7efa67b099?auto=format&fit=crop&w=800&q=80" alt="Accessories" />
              <div className="cat-content">
                <h3>配件與家居</h3>
                <span>Shop Accessories</span>
              </div>
            </Link>
          </div>
          
          <div className="banner-grid-bottom reveal-on-scroll">
            <Link to="/shop?category=toddler" className="cat-banner tall">
              <img src="https://images.unsplash.com/photo-1519241047957-be31d7379a5d?auto=format&fit=crop&w=800&q=80" alt="Toddler" />
              <div className="cat-content">
                <h3>小童穿搭 (3-6歲)</h3>
                <span>Shop Toddler</span>
              </div>
            </Link>
            <Link to="/shop?category=kids" className="cat-banner">
              <img src="https://images.unsplash.com/photo-1622290291468-a28f7a7dc6a8?auto=format&fit=crop&w=1200&q=80" alt="Kids" />
              <div className="cat-content">
                <h3>大童精選 (7-12歲)</h3>
                <span>Shop Kids</span>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* 4. New Arrivals */}
      <section style={{ padding: '0 0 5rem', position: 'relative', zIndex: 1 }}>
        <div className="container reveal-on-scroll" style={{ paddingTop: '5rem' }}>
          <span className="section-subtitle">New Collection</span>
          <h2 className="section-title"><span>最新</span>上架商品</h2>
          <div className="grid grid-4">
            {newArrivals.map(product => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
          <div className="text-center" style={{ marginTop: '3rem' }}>
            <Link to="/shop" className="btn btn-outline" style={{ padding: '0.8rem 3rem' }}>瀏覽全部商品</Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
