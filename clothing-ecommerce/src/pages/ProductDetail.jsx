import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { products } from '../data';
import { ChevronDown, ChevronUp, Star, Truck, RefreshCw, ShieldCheck, Heart } from 'lucide-react';

const ProductDetail = ({ addToCart }) => {
  const { id } = useParams();
  const product = products.find(p => p.id === parseInt(id)) || products[0];
  
  const [activeImage, setActiveImage] = useState(product.images[0]);
  const [size, setSize] = useState(product.sizes[0] || '80cm');
  const [color, setColor] = useState(product.colors[0] || 'cream');
  const [qty, setQty] = useState(1);
  const [openAccordion, setOpenAccordion] = useState('details');

  useEffect(() => {
    window.scrollTo(0, 0);
    setActiveImage(product.images[0]);
    setSize(product.sizes[0] || '80cm');
    setColor(product.colors[0] || 'cream');
  }, [product]);

  const toggleAccordion = (section) => {
    setOpenAccordion(openAccordion === section ? null : section);
  };

  const handleAddToCart = () => {
    addToCart(product, size, qty);
  };

  // Recommendations
  const relatedProducts = products.filter(p => p.id !== product.id && p.category === product.category).slice(0, 4);
  if(relatedProducts.length < 4) {
      relatedProducts.push(...products.filter(p => p.id !== product.id && p.category !== product.category).slice(0, 4 - relatedProducts.length));
  }

  return (
    <div>
      <div className="container pdp-layout">
        
        {/* Left: Images */}
        <div>
          <div className="pdp-gallery">
            <div className="pdp-main-img">
              <img src={activeImage} alt={product.title} />
            </div>
            <div className="pdp-thumbnails">
              {product.images.map((img, idx) => (
                <div 
                  key={idx} 
                  className={`pdp-thumb ${activeImage === img ? 'active' : ''}`}
                  onClick={() => setActiveImage(img)}
                >
                  <img src={img} alt={`Thumbnail ${idx}`} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Info Pro */}
        <div className="pdp-info">
          <div className="pdp-breadcrumbs">
            <Link to="/">Home</Link> &gt; <Link to={`/shop?category=${product.category}`}>Shop {product.category}</Link> &gt; {product.title}
          </div>

          <h1 className="pdp-title" style={{ fontFamily: 'var(--font-serif)' }}>{product.title}</h1>
          
          <div className="pdp-price-wrap">
            <div className="pdp-price">
              {product.originalPrice && <span style={{ textDecoration: 'line-through', color: '#aaa', marginRight: '15px', fontSize: '1.2rem', fontWeight: '400' }}>NT$ {product.originalPrice}</span>}
              <span style={{ color: product.isSale ? 'var(--color-error)' : 'var(--color-primary-dark)' }}>NT$ {product.price}</span>
            </div>
            <div className="pdp-reviews">
              <Star size={16} fill="currentColor" />
              <Star size={16} fill="currentColor" />
              <Star size={16} fill="currentColor" />
              <Star size={16} fill="currentColor" />
              <Star size={16} fill="currentColor" />
              <span style={{ color: 'var(--color-text-light)', fontSize: '0.9rem', marginLeft: '5px' }}>{product.rating} ({product.reviews} reviews)</span>
            </div>
          </div>

          <p className="pdp-desc">
            {product.description}
          </p>

          <div style={{ marginBottom: '2.5rem' }}>
            <div className="selector-title">
              <span>顏色 Color - {color.toUpperCase()}</span>
            </div>
            <div className="color-selector">
              {product.colors.includes('cream') && <button className={`color-opt ${color === 'cream' ? 'active' : ''}`} style={{ backgroundColor: '#fffddd' }} onClick={() => setColor('cream')}></button>}
              {product.colors.includes('oatmeal') && <button className={`color-opt ${color === 'oatmeal' ? 'active' : ''}`} style={{ backgroundColor: '#e8ddcb' }} onClick={() => setColor('oatmeal')}></button>}
              {product.colors.includes('denim') && <button className={`color-opt ${color === 'denim' ? 'active' : ''}`} style={{ backgroundColor: '#4b7596' }} onClick={() => setColor('denim')}></button>}
              {product.colors.includes('olive') && <button className={`color-opt ${color === 'olive' ? 'active' : ''}`} style={{ backgroundColor: '#757c61' }} onClick={() => setColor('olive')}></button>}
              {product.colors.includes('navy') && <button className={`color-opt ${color === 'navy' ? 'active' : ''}`} style={{ backgroundColor: '#1d2738' }} onClick={() => setColor('navy')}></button>}
              {product.colors.includes('white') && <button className={`color-opt ${color === 'white' ? 'active' : ''}`} style={{ backgroundColor: '#ffffff' }} onClick={() => setColor('white')}></button>}
            </div>

            <div className="selector-title">
              <span>尺寸 Size</span>
              <a href="#" style={{ textDecoration: 'underline', color: 'var(--color-text-light)' }}>查看尺寸表</a>
            </div>
            <div className="size-selector">
              {product.sizes.map(s => (
                <button 
                  key={s} 
                  className={`size-opt ${size === s ? 'active' : ''}`}
                  onClick={() => setSize(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="add-block">
            <div className="qty-box">
              <button onClick={() => setQty(Math.max(1, qty - 1))}>-</button>
              <input type="number" value={qty} readOnly />
              <button onClick={() => setQty(qty + 1)}>+</button>
            </div>
            <button className="btn btn-add-cart btn-primary" onClick={handleAddToCart}>加入購物車</button>
            <button className="icon-btn" style={{ border: '1px solid var(--color-border)', borderRadius: '4px', padding: '0 15px' }}><Heart size={22} /></button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1.5rem', backgroundColor: '#fcfcfc', borderRadius: '6px', marginBottom: '3rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.9rem', color: 'var(--color-text)' }}>
              <Truck size={18} color="var(--color-primary)" /> 全館消費滿 NT$1,500 享免運費 (限台灣本島)
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.9rem', color: 'var(--color-text)' }}>
              <RefreshCw size={18} color="var(--color-primary)" /> 享有 14 天無條件退換貨服務（貼身衣物除外）
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.9rem', color: 'var(--color-text)' }}>
              <ShieldCheck size={18} color="var(--color-primary)" /> 通過 Oeko-Tex 100 第一級嬰幼兒最高環保認證
            </div>
          </div>

          <div className="accordion-wrapper">
            <div className="accordion">
              <button className="accordion-btn" onClick={() => toggleAccordion('details')}>
                商品規格 <ChevronDown size={20} style={{ transform: openAccordion === 'details' ? 'rotate(180deg)' : 'none', transition: '0.3s' }}/>
              </button>
              <div className={`accordion-content ${openAccordion === 'details' ? 'open' : ''}`}>
                <ul>
                  <li>材質：100% GOTS 有機棉</li>
                  <li>厚度：適中（適合春夏秋三季）</li>
                  <li>產地：台灣設計，在地工廠製作</li>
                  <li>細節：領口特殊防刮設計、開檔隱形壓扣</li>
                </ul>
              </div>
            </div>
            <div className="accordion">
              <button className="accordion-btn" onClick={() => toggleAccordion('care')}>
                洗滌與保養方式 <ChevronDown size={20} style={{ transform: openAccordion === 'care' ? 'rotate(180deg)' : 'none', transition: '0.3s' }}/>
              </button>
              <div className={`accordion-content ${openAccordion === 'care' ? 'open' : ''}`}>
                <p>為保護有機棉幼嫩纖維與寶寶肌膚，建議：</p>
                <ul>
                  <li>使用中性天然洗劑，水溫不超過30度。</li>
                  <li>反面套入洗衣袋後以冷水低速機洗。</li>
                  <li>不可使用漂白水、螢光增白劑。</li>
                  <li>請陰涼處平放晾乾，不可高溫烘乾避免縮水。</li>
                </ul>
              </div>
            </div>
            <div className="accordion">
              <button className="accordion-btn" onClick={() => toggleAccordion('shipping')}>
                運送與退貨政策 <ChevronDown size={20} style={{ transform: openAccordion === 'shipping' ? 'rotate(180deg)' : 'none', transition: '0.3s' }}/>
              </button>
              <div className={`accordion-content ${openAccordion === 'shipping' ? 'open' : ''}`}>
                <p>現貨商品將於完成付款後 1-3 個工作天內出貨（不含週休及國定假日）。</p>
                <p>若收到商品有瑕疵或尺寸不合，請於收到商品後 14 天內保持商品全新未下水狀態，聯繫客服辦理退換貨。</p>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Related Products */}
      <section style={{ padding: '4rem 0 6rem', backgroundColor: '#fcfbf9' }}>
        <div className="container">
          <h2 className="section-title" style={{ fontSize: '1.8rem', marginBottom: '2.5rem' }}>您可能也會喜歡</h2>
          <div className="grid grid-4">
            {relatedProducts.map(product => (
              <div key={product.id} className="product-card">
                <div className="product-image-wrap">
                  <Link to={`/product/${product.id}`}>
                    <img src={product.images[0]} alt={product.title} className="product-image primary-img" />
                    {product.images[1] && <img src={product.images[1]} alt={product.title} className="product-image hover-img" />}
                  </Link>
                </div>
                <div className="product-info">
                  <h3 className="product-title"><Link to={`/product/${product.id}`}>{product.title}</Link></h3>
                  <div className="product-price">NT$ {product.price}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default ProductDetail;
